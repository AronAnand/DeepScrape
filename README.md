# DeepScrape

FastAPI web scraper that automatically detects containers with links and provides deep website scraping capabilities.

## Features

- **Real-time Progress Logging**: Shows "Found URL" and "Scraping URL" messages in console
- **Two Scraping Modes**: Single page vs deep website scraping
- **Automatic Container Detection**: Finds div elements with IDs that contain links (no need to specify container IDs)
- **Multiple Output Formats**: JSON responses and downloadable DOCX files
- **Error Handling**: Graceful handling of failed requests with detailed logging

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Single Scrape (Single URL Only)
```bash
POST /single-scrape
```
**Body:**
```json
{
  "url": "https://www.lockhartisd.org/"
}
```

**Response:** Scraped content from just the single URL

### 2. Deep Scrape (Full Website)
```bash
POST /deep-scrape
```
**Body:**
```json
{
  "url": "https://www.lockhartisd.org/"
}
```

**Response:** 
- Main page content
- All linked pages content
- Statistics about successful/failed scrapes
- Detected container IDs

**Console Output Shows:**
- `Starting deep scrape for: <url>`
- `Analyzing main page: <url>`
- `Detected containers: <container_ids>`
- `Found <number> unique URLs`
- `Found URL [1/50]: <url>`
- `Scraping URL [1/50]: <url>`
- `✓ Successfully scraped: <url>`
- `✗ Failed to scrape: <url> - <error>`
- `Deep scrape completed. Success: X, Failed: Y`

### 3. Download Single Scrape as DOCX
```bash
POST /single-scrape-docx
```

### 4. Download Deep Scrape as DOCX
```bash
POST /deep-scrape-docx
```

## Container Detection Logic

The scraper automatically detects containers in this order:

1. **DIV elements with IDs** that contain links
2. **Semantic elements** (`main`, `section`, `article`, `nav`) with IDs that contain links  
3. **Fallback**: Element with the most links
4. **Last resort**: All links from the entire page (using "body" as container ID)

## Example Usage

```bash
# Single page scraping
curl -X POST "http://localhost:8001/single-scrape" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.lockhartisd.org/"}'

# Deep scraping (watch console for progress)
curl -X POST "http://localhost:8001/deep-scrape" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.lockhartisd.org/"}'

# Download deep scrape as DOCX
curl -X POST "http://localhost:8001/deep-scrape-docx" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.lockhartisd.org/"}' \
  --output deep_scrape_results.docx
```

## API Documentation

Visit `http://localhost:8001/docs` for interactive API documentation (Swagger UI).

## Results Display and Downloads

After scraping completes, the web interface shows:

### **📊 Results Section**
- **Summary Statistics**: URLs found, success/failure counts, detected containers
- **Content Preview**: Title and first 500 characters of each scraped page
- **Full Content**: Scrollable view of all scraped text

### **📥 Download Options**
- **JSON Download**: Complete structured data with all scraped content
- **DOCX Download**: Formatted document with all results

## Example Results Display

```
📈 Summary:
🌍 Main URL: https://example.com/
📄 Total Pages: 25
✅ Successful: 23
❌ Failed: 2
📦 Detected Containers: fsPageWrapper, mainContent

🔗 https://example.com/
📝 Home - Example Website
Lorem ipsum dolor sit amet, consectetur adipiscing elit...

🔗 https://example.com/about
📝 About Us - Example Website  
We are a leading company in our field, established in...
```

## Notes

- The scraper respects timeouts (30s for main page, 20s for each linked page)
- Failed page requests are logged but don't stop the scraping process
- Large sites may take several minutes to fully scrape
- All scraped text is cleaned (scripts, styles, etc. removed)
- **Results are displayed immediately** after scraping completes
- **Downloads work for both single and deep scraping modes**