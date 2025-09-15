# -*- coding: utf-8 -*-
"""
FastAPI Web Scraper
Scrapes websites and automatically detects containers with links
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Generator
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
import json
import re
import os
import logging
import sys
from pathlib import Path
from docx import Document
import tempfile
import uuid
from datetime import datetime
import threading
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Web Scraper API", description="API for scraping website content with real-time progress")

# Global progress storage
progress_store = {}

class ScrapeRequest(BaseModel):
    url: HttpUrl

class ScrapedLink(BaseModel):
    container_id: str
    anchor_text: Optional[str]
    url: str

class ScrapedContent(BaseModel):
    url: str
    container_id: Optional[str]
    anchor_text: Optional[str]
    title: str
    text: str

def absolutize(href: str, base: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith("//"):
        scheme = urlsplit(base).scheme or "https"
        return f"{scheme}:{href}"
    return urljoin(base, href)

def visible_text_only(soup_obj: BeautifulSoup) -> str:
    """Extract clean, visible text content from BeautifulSoup object"""
    # Remove unwanted tags
    for tag in soup_obj(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    
    # Get text with proper spacing
    text = soup_obj.get_text("\n", strip=True)
    
    # Clean up excessive whitespace while preserving structure
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    
    return text.strip()

def clean_text(html: str) -> str:
    """Enhanced text extraction similar to the original scraper"""
    soup = BeautifulSoup(html, "lxml")
    
    # Remove unwanted elements (more comprehensive)
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "button", "input", "form"]):
        tag.decompose()
    
    # Extract main content areas first
    main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': re.compile(r'content|main|body', re.I)}) or soup
    
    # Get clean text
    text = main_content.get_text("\n", strip=True)
    
    # Advanced text cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # Remove excess spaces/tabs
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # Trim line whitespace
    
    return text.strip()

def detect_containers_with_links(soup: BeautifulSoup) -> List[str]:
    """Automatically detect containers (divs with IDs) that contain links"""
    containers = []
    
    # Find all divs with IDs that contain links
    for div in soup.find_all('div', id=True):
        links = div.find_all('a', href=True)
        if links:  # If this div contains links
            containers.append(div.get('id'))
    
    # If no divs with IDs found, look for other containers
    if not containers:
        for tag_name in ['main', 'section', 'article', 'nav']:
            for element in soup.find_all(tag_name, id=True):
                links = element.find_all('a', href=True)
                if links:
                    containers.append(element.get('id'))
    
    # Fallback: if still no containers, use body or any element with most links
    if not containers:
        all_elements_with_id = soup.find_all(id=True)
        if all_elements_with_id:
            # Find element with most links
            best_element = max(all_elements_with_id, 
                             key=lambda x: len(x.find_all('a', href=True)))
            if best_element.find_all('a', href=True):
                containers.append(best_element.get('id'))
    
    return containers

def extract_all_links(url: str, log_callback=None) -> List[ScrapedLink]:
    try:
        if log_callback:
            log_callback(f"Analyzing main page: {url}")
        
        resp = requests.get(
            str(url), 
            timeout=30, 
            headers={"User-Agent": "Mozilla/5.0 (compatible; FastAPIScraper/1.0)"}
        )
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "lxml")
        
        # Auto-detect containers with links
        container_ids = detect_containers_with_links(soup)
        
        if log_callback and container_ids:
            log_callback(f"Detected containers: {', '.join(container_ids)}")
        
        extracted = []
        
        if container_ids:
            # Extract from detected containers
            for cid in container_ids:
                container = soup.find(id=cid)
                if container:
                    for a in container.select("a[href]"):
                        link_url = absolutize(a.get("href"), str(url))
                        if link_url:
                            extracted.append(ScrapedLink(
                                container_id=cid,
                                anchor_text=a.get_text(strip=True) or None,
                                url=link_url
                            ))
        else:
            # Fallback: extract all links from page
            for a in soup.select("a[href]"):
                link_url = absolutize(a.get("href"), str(url))
                if link_url:
                    extracted.append(ScrapedLink(
                        container_id="body",
                        anchor_text=a.get_text(strip=True) or None,
                        url=link_url
                    ))
        
        # Deduplicate by URL
        seen = set()
        deduped = []
        for item in extracted:
            if item.url not in seen:
                seen.add(item.url)
                deduped.append(item)
        
        if log_callback:
            log_callback(f"Found {len(deduped)} unique URLs")
        
        return deduped
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting links: {str(e)}")

def scrape_url_content(url: str, log_callback=None) -> ScrapedContent:
    try:
        if log_callback:
            log_callback(f"Scraping URL: {url}")
        
        resp = requests.get(
            url, 
            timeout=20, 
            headers={"User-Agent": "Mozilla/5.0 (FastAPIContentFetcher/1.0)"}
        )
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = clean_text(resp.text)
        
        if log_callback:
            log_callback(f"✓ Successfully scraped: {url}")
        
        return ScrapedContent(
            url=url,
            container_id=None,
            anchor_text=None,
            title=title,
            text=text
        )
    
    except Exception as e:
        if log_callback:
            log_callback(f"✗ Failed to scrape: {url} - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error scraping {url}: {str(e)}")

def convert_to_docx(data: List[ScrapedContent], output_path: str) -> None:
    doc = Document()
    for item in data:
        json_line = json.dumps(item.dict(), ensure_ascii=False)
        doc.add_paragraph(json_line)
    doc.save(output_path)

async def scrape_url_async(session: aiohttp.ClientSession, url: str) -> ScrapedContent:
    """Async version of URL scraping"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "lxml")
                title = soup.title.get_text(strip=True) if soup.title else ""
                text = clean_text(html)
                
                return ScrapedContent(
                    url=url,
                    container_id=None,
                    anchor_text=None,
                    title=title,
                    text=text
                )
            else:
                raise Exception(f"HTTP {response.status}")
    except Exception as e:
        raise Exception(f"Error scraping {url}: {str(e)}")

async def scrape_urls_batch(urls: List[str], batch_size: int = 10) -> List[ScrapedContent]:
    """Scrape multiple URLs concurrently in batches"""
    results = []
    connector = aiohttp.TCPConnector(limit=batch_size)
    headers = {"User-Agent": "Mozilla/5.0 (FastAPIBatchScraper/1.0)"}
    
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        # Process URLs in batches to avoid overwhelming the server
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            tasks = [scrape_url_async(session, url) for url in batch]
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.info(f"✗ Failed to scrape: {batch[j]} - {str(result)}")
                else:
                    results.append(result)
                    logger.info(f"✓ Successfully scraped: {batch[j]}")
            
            # Small delay between batches to be respectful
            if i + batch_size < len(urls):
                await asyncio.sleep(0.1)
    
    return results

@app.get("/")
async def root():
    return {"message": "Web Scraper API is running"}

@app.get("/stream-viewer", response_class=HTMLResponse)
async def stream_viewer():
    """Serve the streaming progress viewer HTML page"""
    try:
        with open("stream_viewer.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Stream viewer not found</h1><p>Please ensure stream_viewer.html exists in the same directory as main.py</p>",
            status_code=404
        )

@app.post("/test-results")
async def test_results():
    """Test endpoint to verify results display"""
    test_data = {
        "main_url": "https://test.com",
        "total_pages_scraped": 2,
        "successful_scrapes": 2,
        "failed_scrapes": 0,
        "detected_containers": ["testContainer"],
        "results": [
            {
                "url": "https://test.com",
                "title": "Test Page 1",
                "text": "This is test content for page 1. Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "container_id": None,
                "anchor_text": None
            },
            {
                "url": "https://test.com/page2", 
                "title": "Test Page 2",
                "text": "This is test content for page 2. Sed do eiusmod tempor incididunt ut labore.",
                "container_id": "testContainer",
                "anchor_text": "Page 2"
            }
        ]
    }
    return test_data

@app.post("/single-scrape", response_model=ScrapedContent)
async def single_scrape(request: ScrapeRequest):
    """Scrape content from a single URL only (no deep linking)"""
    logger.info(f"Starting single scrape for: {request.url}")
    
    def log_progress(message: str):
        logger.info(message)
    
    result = scrape_url_content(str(request.url), log_progress)
    logger.info(f"Single scrape completed for: {request.url}")
    return result

@app.post("/deep-scrape")
async def deep_scrape(request: ScrapeRequest):
    """Deep scrape: main page and all linked pages with progress logging"""
    logger.info(f"Starting deep scrape for: {request.url}")
    
    def log_progress(message: str):
        logger.info(message)
    
    # Extract links first
    links = extract_all_links(str(request.url), log_progress)
    
    # Get main page content
    main_content = scrape_url_content(str(request.url), log_progress)
    
    # Scrape all linked pages
    results = [main_content]
    successful_scrapes = 1
    failed_scrapes = 0
    
    for i, link in enumerate(links, 1):
        logger.info(f"Found URL [{i}/{len(links)}]: {link.url}")
        logger.info(f"Scraping URL [{i}/{len(links)}]: {link.url}")
        try:
            content = scrape_url_content(link.url, log_progress)
            content.container_id = link.container_id
            content.anchor_text = link.anchor_text
            results.append(content)
            successful_scrapes += 1
        except Exception as e:
            logger.info(f"✗ Failed to scrape: {link.url} - {str(e)}")
            failed_scrapes += 1
            continue
    
    logger.info(f"Deep scrape completed. Success: {successful_scrapes}, Failed: {failed_scrapes}")
    
    return {
        "main_url": str(request.url),
        "total_links_found": len(links),
        "successful_scrapes": successful_scrapes,
        "failed_scrapes": failed_scrapes,
        "detected_containers": list(set([link.container_id for link in links])),
        "results": results
    }

@app.post("/deep-scrape-with-progress")
async def deep_scrape_with_progress(request: ScrapeRequest):
    """Deep scrape with detailed progress updates stored for polling"""
    job_id = str(uuid.uuid4())[:8]
    progress_store[job_id] = {
        "status": "starting",
        "messages": [],
        "current_step": 0,
        "total_steps": 0,
        "results": None
    }
    
    def add_progress(message: str, msg_type: str = "progress"):
        progress_store[job_id]["messages"].append({
            "message": message,
            "type": msg_type,
            "timestamp": datetime.now().isoformat()
        })
    
    def run_scraping():
        try:
            add_progress(f"🚀 Starting deep scrape for: {request.url}", "start")
            add_progress(f"🔍 Analyzing main page: {request.url}", "progress")
            
            # Extract links first
            links = extract_all_links(str(request.url))
            
            if links:
                add_progress(f"✅ Detected containers and found {len(links)} unique URLs", "info")
            else:
                add_progress(f"📊 No links found, scraping single page only", "info")
            
            progress_store[job_id]["total_steps"] = len(links) + 1
            
            # Get main page content
            add_progress(f"📄 Scraping main page: {request.url}", "scraping")
            main_content = scrape_url_content(str(request.url))
            add_progress(f"✅ Successfully scraped main page", "success")
            
            results = [main_content]
            successful_scrapes = 1
            failed_scrapes = 0
            progress_store[job_id]["current_step"] = 1
            
            # Scrape all linked pages
            for i, link in enumerate(links, 1):
                add_progress(f"🔗 Found URL [{i}/{len(links)}]: {link.url}", "found")
                add_progress(f"🌐 Scraping URL [{i}/{len(links)}]: {link.url}", "scraping")
                
                try:
                    content = scrape_url_content(link.url)
                    content.container_id = link.container_id
                    content.anchor_text = link.anchor_text
                    results.append(content)
                    successful_scrapes += 1
                    add_progress(f"✅ Successfully scraped [{i}/{len(links)}]: {link.url}", "success")
                except Exception as e:
                    failed_scrapes += 1
                    add_progress(f"❌ Failed to scrape [{i}/{len(links)}]: {link.url} - {str(e)}", "error")
                
                progress_store[job_id]["current_step"] = i + 1
            
            final_result = {
                "main_url": str(request.url),
                "total_links_found": len(links),
                "successful_scrapes": successful_scrapes,
                "failed_scrapes": failed_scrapes,
                "detected_containers": list(set([link.container_id for link in links])) if links else [],
                "results": results
            }
            
            add_progress(f"🎉 Deep scrape completed! ✅ Success: {successful_scrapes}, ❌ Failed: {failed_scrapes}", "complete")
            progress_store[job_id]["status"] = "completed"
            progress_store[job_id]["results"] = final_result
            
        except Exception as e:
            add_progress(f"💥 Critical error: {str(e)}", "error")
            progress_store[job_id]["status"] = "error"
    
    # Run scraping in background thread
    threading.Thread(target=run_scraping, daemon=True).start()
    
    return {"job_id": job_id, "status": "started"}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Get progress updates for a scraping job"""
    if job_id not in progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return progress_store[job_id]

@app.post("/deep-scrape-batch")
async def deep_scrape_batch(request: ScrapeRequest, batch_size: int = Query(10)):
    """Fast batch deep scraping with concurrent processing"""
    job_id = str(uuid.uuid4())[:8]
    progress_store[job_id] = {
        "status": "starting",
        "messages": [],
        "current_step": 0,
        "total_steps": 0,
        "results": None
    }
    
    def add_progress(message: str, msg_type: str = "progress"):
        progress_store[job_id]["messages"].append({
            "message": message,
            "type": msg_type,
            "timestamp": datetime.now().isoformat()
        })
    
    async def run_batch_scraping():
        try:
            add_progress(f"🚀 Starting BATCH deep scrape for: {request.url}", "start")
            add_progress(f"⚡ Using concurrent processing with batch size: {batch_size}", "info")
            add_progress(f"🔍 Analyzing main page: {request.url}", "progress")
            
            # Extract links first (synchronous)
            links = extract_all_links(str(request.url))
            
            if links:
                add_progress(f"✅ Found {len(links)} URLs for batch processing", "info")
            else:
                add_progress(f"📊 No links found, scraping single page only", "info")
            
            progress_store[job_id]["total_steps"] = len(links) + 1
            
            # Get main page content (synchronous)
            add_progress(f"📄 Scraping main page: {request.url}", "scraping")
            main_content = scrape_url_content(str(request.url))
            add_progress(f"✅ Successfully scraped main page", "success")
            
            results = [main_content]
            progress_store[job_id]["current_step"] = 1
            
            if links:
                # Batch scrape all linked pages CONCURRENTLY
                urls_to_scrape = [link.url for link in links]
                add_progress(f"⚡ Starting batch processing of {len(urls_to_scrape)} URLs...", "info")
                add_progress(f"🔄 Processing {batch_size} URLs simultaneously per batch", "info")
                
                # Run async batch scraping
                batch_results = await scrape_urls_batch(urls_to_scrape, batch_size)
                
                # Match results back to original links for metadata
                successful_scrapes = 1  # main page
                failed_scrapes = 0
                
                for i, link in enumerate(links):
                    # Find matching result
                    matching_result = None
                    for result in batch_results:
                        if result.url == link.url:
                            matching_result = result
                            break
                    
                    if matching_result:
                        matching_result.container_id = link.container_id
                        matching_result.anchor_text = link.anchor_text
                        results.append(matching_result)
                        successful_scrapes += 1
                        add_progress(f"✅ [{successful_scrapes-1}/{len(links)}] Completed: {link.url}", "success")
                    else:
                        failed_scrapes += 1
                        add_progress(f"❌ [{successful_scrapes+failed_scrapes-1}/{len(links)}] Failed: {link.url}", "error")
                    
                    progress_store[job_id]["current_step"] = i + 2
                
                add_progress(f"⚡ Batch processing completed! Much faster than sequential!", "info")
            else:
                successful_scrapes = 1
                failed_scrapes = 0
            
            final_result = {
                "main_url": str(request.url),
                "total_links_found": len(links),
                "successful_scrapes": successful_scrapes,
                "failed_scrapes": failed_scrapes,
                "detected_containers": list(set([link.container_id for link in links])) if links else [],
                "results": results,
                "batch_size_used": batch_size,
                "processing_type": "concurrent_batch"
            }
            
            add_progress(f"🎉 BATCH deep scrape completed! ✅ Success: {successful_scrapes}, ❌ Failed: {failed_scrapes}", "complete")
            add_progress(f"⚡ Speed boost: Processed {len(links)} URLs concurrently in batches!", "complete")
            progress_store[job_id]["status"] = "completed"
            progress_store[job_id]["results"] = final_result
            
        except Exception as e:
            add_progress(f"💥 Critical error: {str(e)}", "error")
            progress_store[job_id]["status"] = "error"
    
    # Run batch scraping in background
    asyncio.create_task(run_batch_scraping())
    
    return {"job_id": job_id, "status": "started", "batch_size": batch_size, "mode": "concurrent_batch"}

@app.post("/single-scrape-docx")
async def single_scrape_docx(request: ScrapeRequest):
    """Scrape single URL content and return as downloadable DOCX file"""
    logger.info(f"Generating DOCX for single scrape: {request.url}")
    
    def log_progress(message: str):
        logger.info(message)
    
    # Get single URL content
    content = scrape_url_content(str(request.url), log_progress)
    results = [content]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    temp_file.close()
    
    # Convert to DOCX
    convert_to_docx(results, temp_file.name)
    
    return FileResponse(
        temp_file.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"single_scrape_{uuid.uuid4().hex[:8]}.docx"
    )

@app.post("/deep-scrape-docx")
async def deep_scrape_docx(request: ScrapeRequest):
    """Deep scrape all content and return as downloadable DOCX file"""
    logger.info(f"Generating DOCX for deep scrape: {request.url}")
    
    # Get all content using deep scrape
    scrape_result = await deep_scrape(request)
    results = scrape_result["results"]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    temp_file.close()
    
    # Convert to DOCX
    convert_to_docx(results, temp_file.name)
    
    return FileResponse(
        temp_file.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"deep_scrape_{uuid.uuid4().hex[:8]}.docx"
    )

@app.post("/single-scrape-stream")
async def single_scrape_stream(request: ScrapeRequest):
    """Single scrape with real-time progress streaming (visible in browser)"""
    
    def generate_progress() -> Generator[str, None, None]:
        try:
            yield f"data: {{\"message\": \"🚀 Starting single scrape for: {request.url}\", \"type\": \"start\"}}\n\n"
            yield f"data: {{\"message\": \"🔍 Analyzing single page: {request.url}\", \"type\": \"progress\"}}\n\n"
            
            try:
                resp = requests.get(
                    str(request.url), 
                    timeout=30, 
                    headers={"User-Agent": "Mozilla/5.0 (compatible; FastAPIScraper/1.0)"}
                )
                resp.raise_for_status()
                
                yield f"data: {{\"message\": \"📄 Scraping URL: {request.url}\", \"type\": \"scraping\"}}\n\n"
                
                soup = BeautifulSoup(resp.text, "lxml")
                title = soup.title.get_text(strip=True) if soup.title else ""
                text = clean_text(resp.text)
                
                result = ScrapedContent(
                    url=str(request.url),
                    container_id=None,
                    anchor_text=None,
                    title=title,
                    text=text
                )
                
                yield f"data: {{\"message\": \"✅ Successfully scraped: {request.url}\", \"type\": \"success\"}}\n\n"
                
                final_result = {
                    "main_url": str(request.url),
                    "total_pages_scraped": 1,
                    "successful_scrapes": 1,
                    "failed_scrapes": 0,
                    "results": [result.dict()]
                }
                
                yield f"data: {{\"message\": \"🎉 Single scrape completed! ✅ Success: 1, ❌ Failed: 0\", \"type\": \"complete\", \"final_stats\": {{\"successful\": 1, \"failed\": 0, \"total\": 1}}}}\n\n"
                yield f"data: {{\"type\": \"final_result\", \"data\": {json.dumps(final_result)}}}\n\n"
                
            except Exception as e:
                yield f"data: {{\"message\": \"💥 Error during scraping: {str(e)}\", \"type\": \"error\"}}\n\n"
            
        except Exception as e:
            yield f"data: {{\"message\": \"💥 Critical error: {str(e)}\", \"type\": \"error\"}}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/deep-scrape-stream")
async def deep_scrape_stream(request: ScrapeRequest):
    """Deep scrape with real-time progress streaming (visible in browser)"""
    
    def generate_progress() -> Generator[str, None, None]:
        try:
            yield f"data: {{\"message\": \"🚀 Starting deep scrape for: {request.url}\", \"type\": \"start\"}}\n\n"
            
            # Extract links first
            def log_to_stream(message: str):
                return f"data: {{\"message\": \"{message}\", \"type\": \"progress\"}}\n\n"
            
            yield f"data: {{\"message\": \"🔍 Analyzing main page: {request.url}\", \"type\": \"progress\"}}\n\n"
            
            try:
                resp = requests.get(
                    str(request.url), 
                    timeout=30, 
                    headers={"User-Agent": "Mozilla/5.0 (compatible; FastAPIScraper/1.0)"}
                )
                resp.raise_for_status()
                
                soup = BeautifulSoup(resp.text, "lxml")
                container_ids = detect_containers_with_links(soup)
                
                if container_ids:
                    yield f"data: {{\"message\": \"✅ Detected containers: {', '.join(container_ids)}\", \"type\": \"info\"}}\n\n"
                
                extracted = []
                
                if container_ids:
                    for cid in container_ids:
                        container = soup.find(id=cid)
                        if container:
                            for a in container.select("a[href]"):
                                link_url = absolutize(a.get("href"), str(request.url))
                                if link_url:
                                    extracted.append(ScrapedLink(
                                        container_id=cid,
                                        anchor_text=a.get_text(strip=True) or None,
                                        url=link_url
                                    ))
                else:
                    for a in soup.select("a[href]"):
                        link_url = absolutize(a.get("href"), str(request.url))
                        if link_url:
                            extracted.append(ScrapedLink(
                                container_id="body",
                                anchor_text=a.get_text(strip=True) or None,
                                url=link_url
                            ))
                
                # Deduplicate by URL
                seen = set()
                links = []
                for item in extracted:
                    if item.url not in seen:
                        seen.add(item.url)
                        links.append(item)
                
                yield f"data: {{\"message\": \"📊 Found {len(links)} unique URLs to scrape\", \"type\": \"info\"}}\n\n"
                
                # Get main page content
                yield f"data: {{\"message\": \"📄 Scraping main page: {request.url}\", \"type\": \"scraping\"}}\n\n"
                main_content = scrape_url_content(str(request.url))
                yield f"data: {{\"message\": \"✅ Successfully scraped main page\", \"type\": \"success\"}}\n\n"
                
                results = [main_content]
                successful = 1
                failed = 0
                
                for i, link in enumerate(links, 1):
                    yield f"data: {{\"message\": \"🔗 Found URL [{i}/{len(links)}]: {link.url}\", \"type\": \"found\"}}\n\n"
                    yield f"data: {{\"message\": \"🌐 Scraping URL [{i}/{len(links)}]: {link.url}\", \"type\": \"scraping\"}}\n\n"
                    
                    try:
                        content = scrape_url_content(link.url)
                        content.container_id = link.container_id
                        content.anchor_text = link.anchor_text
                        results.append(content)
                        successful += 1
                        yield f"data: {{\"message\": \"✅ Successfully scraped [{i}/{len(links)}]: {link.url}\", \"type\": \"success\"}}\n\n"
                    except Exception as e:
                        failed += 1
                        yield f"data: {{\"message\": \"❌ Failed to scrape [{i}/{len(links)}]: {link.url} - {str(e)}\", \"type\": \"error\"}}\n\n"
                
                final_result = {
                    "main_url": str(request.url),
                    "total_links_found": len(links),
                    "successful_scrapes": successful,
                    "failed_scrapes": failed,
                    "detected_containers": list(set([link.container_id for link in links])),
                    "results": [r.dict() for r in results]
                }
                
                yield f"data: {{\"message\": \"🎉 Deep scrape completed! ✅ Success: {successful}, ❌ Failed: {failed}\", \"type\": \"complete\", \"final_stats\": {{\"successful\": {successful}, \"failed\": {failed}, \"total\": {len(links) + 1}}}}}\n\n"
                yield f"data: {{\"type\": \"final_result\", \"data\": {json.dumps(final_result)}}}\n\n"
                
            except Exception as e:
                yield f"data: {{\"message\": \"💥 Error during scraping: {str(e)}\", \"type\": \"error\"}}\n\n"
            
        except Exception as e:
            yield f"data: {{\"message\": \"💥 Critical error: {str(e)}\", \"type\": \"error\"}}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)