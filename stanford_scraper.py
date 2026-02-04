# stanford_scraper.py

import requests
from bs4 import BeautifulSoup
from extractous import Extractor
from pathlib import Path
import time
import json
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import re

class StanfordMedicine25Scraper:
    """
    Comprehensive scraper for Stanford Medicine 25 physical exam content.
    """
    
    def __init__(self, output_dir: str = "StanfordPE"):
        self.base_url = "https://stanfordmedicine25.stanford.edu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.extractor = Extractor()
        self.scraped_urls = set()
        self.failed_urls = []
        
    def discover_exam_pages(self) -> List[Dict[str, str]]:
        """
        Discover all physical exam pages on Stanford Medicine 25.
        
        Returns:
            List of dictionaries with page info (title, url, category)
        """
        print("Discovering physical exam pages...")
        
        # Main pages to check
        discovery_urls = [
            f"{self.base_url}/the25.html",  # Main "The 25" page
            f"{self.base_url}/skills.html",  # Skills page
            f"{self.base_url}/",  # Homepage
        ]
        
        exam_pages = []
        
        for url in discovery_urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all links that might be physical exam pages
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(self.base_url, href)
                    
                    # Filter for relevant pages
                    if self._is_exam_page(full_url) and full_url not in self.scraped_urls:
                        title = link.get_text(strip=True) or self._extract_title_from_url(full_url)
                        category = self._categorize_exam(title, full_url)
                        
                        exam_pages.append({
                            'title': title,
                            'url': full_url,
                            'category': category
                        })
                        self.scraped_urls.add(full_url)
                
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                print(f"Error discovering from {url}: {e}")
        
        # Remove duplicates
        unique_pages = {page['url']: page for page in exam_pages}
        exam_pages = list(unique_pages.values())
        
        print(f"Discovered {len(exam_pages)} physical exam pages")
        return exam_pages
    
    def _is_exam_page(self, url: str) -> bool:
        """Check if URL is likely a physical exam page."""
        # Must be from Stanford Medicine 25
        if 'stanfordmedicine25.stanford.edu' not in url:
            return False
        
        # Exclude certain pages
        exclude_patterns = [
            'about', 'contact', 'faculty', 'news', 'events',
            'blog', 'donate', 'search', 'sitemap', 'privacy',
            '.pdf', '.jpg', '.png', '.mp4', 'video', 'audio'
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in exclude_patterns):
            return False
        
        # Include pages with these patterns
        include_patterns = [
            '/the25/', 'exam', 'maneuver', 'physical', 'clinical',
            'auscultation', 'palpation', 'inspection', 'percussion'
        ]
        
        return any(pattern in url_lower for pattern in include_patterns)
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL."""
        path = urlparse(url).path
        filename = Path(path).stem
        # Convert camelCase or snake_case to Title Case
        title = re.sub(r'([a-z])([A-Z])', r'\1 \2', filename)
        title = title.replace('_', ' ').replace('-', ' ')
        return title.title()
    
    def _categorize_exam(self, title: str, url: str) -> str:
        """Categorize the physical exam by body system."""
        text = (title + " " + url).lower()
        
        categories = {
            'Cardiovascular': ['heart', 'cardiac', 'aortic', 'murmur', 'pulse', 'jvp', 'jugular'],
            'Respiratory': ['lung', 'respiratory', 'breath', 'pneumonia', 'copd', 'asthma'],
            'Neurological': ['neuro', 'reflex', 'cranial', 'cerebellar', 'gait', 'sensory', 'motor'],
            'Musculoskeletal': ['joint', 'bone', 'muscle', 'shoulder', 'knee', 'hip', 'hand', 'foot', 'ankle'],
            'Abdominal': ['abdom', 'liver', 'spleen', 'ascites', 'bowel'],
            'Dermatological': ['skin', 'rash', 'lesion', 'acne', 'mole', 'derma'],
            'HEENT': ['eye', 'ear', 'nose', 'throat', 'thyroid', 'lymph'],
            'General': ['vital', 'appearance', 'nutrition']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'Other'
    
    def scrape_page_with_extractous(self, url: str) -> Optional[Dict[str, any]]:
        """
        Scrape a single page using extractous library.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with content and metadata
        """
        try:
            print(f"Scraping: {url}")
            
            # Extract content using extractous
            reader, metadata = self.extractor.extract_url(url)
            
            # Read content in chunks
            content = ""
            buffer = reader.read(4096)
            while len(buffer) > 0:
                content += buffer.decode("utf-8", errors='ignore')
                buffer = reader.read(4096)
            
            return {
                'url': url,
                'content': content,
                'metadata': metadata,
                'method': 'extractous'
            }
            
        except Exception as e:
            print(f"Extractous failed for {url}: {e}")
            return None
    
    def scrape_page_with_beautifulsoup(self, url: str) -> Optional[Dict[str, any]]:
        """
        Scrape a single page using BeautifulSoup (fallback method).
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with content and metadata
        """
        try:
            print(f"Scraping with BeautifulSoup: {url}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ""
            
            # Get main content
            # Try to find main content area
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=re.compile('content|main|body', re.I)) or
                soup.find('body')
            )
            
            if main_content:
                # Extract text from paragraphs, headings, lists
                content_parts = []
                for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'td']):
                    text = element.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                
                content = "\n".join(content_parts)
            else:
                content = soup.get_text(strip=True)
            
            return {
                'url': url,
                'content': content,
                'metadata': {'title': title_text},
                'method': 'beautifulsoup'
            }
            
        except Exception as e:
            print(f"BeautifulSoup failed for {url}: {e}")
            return None
    
    def scrape_page(self, url: str) -> Optional[Dict[str, any]]:
        """
        Scrape a page using extractous first, fallback to BeautifulSoup.
        """
        # Try extractous first
        result = self.scrape_page_with_extractous(url)
        
        # Fallback to BeautifulSoup if extractous fails
        if not result or not result.get('content'):
            result = self.scrape_page_with_beautifulsoup(url)
        
        return result
    
    def save_content(self, page_info: Dict, scraped_data: Dict) -> str:
        """
        Save scraped content to file.
        
        Args:
            page_info: Page information (title, url, category)
            scraped_data: Scraped content and metadata
            
        Returns:
            Path to saved file
        """
        # Create safe filename
        title = page_info['title']
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        # Create category subdirectory
        category_dir = self.output_dir / page_info['category']
        category_dir.mkdir(exist_ok=True)
        
        # Save text content
        text_file = category_dir / f"{safe_title}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Title: {page_info['title']}\n")
            f.write(f"URL: {page_info['url']}\n")
            f.write(f"Category: {page_info['category']}\n")
            f.write(f"Scraping Method: {scraped_data.get('method', 'unknown')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(scraped_data['content'])
        
        # Save metadata as JSON
        metadata_file = category_dir / f"{safe_title}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'page_info': page_info,
                'metadata': scraped_data.get('metadata', {}),
                'scraping_method': scraped_data.get('method', 'unknown')
            }, f, indent=2)
        
        print(f"Saved: {text_file}")
        return str(text_file)
    
    def scrape_all(self, delay: float = 2.0) -> Dict[str, any]:
        """
        Discover and scrape all physical exam pages.
        
        Args:
            delay: Delay between requests in seconds
            
        Returns:
            Summary of scraping results
        """
        print("=" * 80)
        print("Stanford Medicine 25 Physical Exam Scraper")
        print("=" * 80)
        
        # Discover pages
        exam_pages = self.discover_exam_pages()
        
        if not exam_pages:
            print("No pages discovered. Trying manual list...")
            exam_pages = self._get_manual_page_list()
        
        # Scrape each page
        results = {
            'successful': [],
            'failed': [],
            'total': len(exam_pages)
        }
        
        for i, page_info in enumerate(exam_pages, 1):
            print(f"\n[{i}/{len(exam_pages)}] Processing: {page_info['title']}")
            
            try:
                # Scrape content
                scraped_data = self.scrape_page(page_info['url'])
                
                if scraped_data and scraped_data.get('content'):
                    # Save content
                    saved_file = self.save_content(page_info, scraped_data)
                    results['successful'].append({
                        'page': page_info,
                        'file': saved_file
                    })
                else:
                    results['failed'].append({
                        'page': page_info,
                        'error': 'No content extracted'
                    })
                
                # Be respectful - delay between requests
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error processing {page_info['url']}: {e}")
                results['failed'].append({
                    'page': page_info,
                    'error': str(e)
                })
        
        # Save summary
        summary_file = self.output_dir / "scraping_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SCRAPING SUMMARY")
        print("=" * 80)
        print(f"Total pages: {results['total']}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Success rate: {len(results['successful'])/results['total']*100:.1f}%")
        print(f"\nResults saved to: {self.output_dir}")
        
        return results
    
    def _get_manual_page_list(self) -> List[Dict[str, str]]:
        """
        Manual list of known Stanford Medicine 25 physical exam pages.
        Use this as fallback if automatic discovery fails.
        """
        manual_pages = [
            # Cardiovascular
            {"title": "Aortic Stenosis", "url": f"{self.base_url}/the25/aorticstenosis.html", "category": "Cardiovascular"},
            {"title": "Aortic Regurgitation", "url": f"{self.base_url}/the25/aorticregurgitation.html", "category": "Cardiovascular"},
            {"title": "Cardiac Second Sounds", "url": f"{self.base_url}/the25/s2.html", "category": "Cardiovascular"},
            {"title": "Diastolic Murmurs", "url": f"{self.base_url}/the25/dm.html", "category": "Cardiovascular"},
            {"title": "JVP Examination", "url": f"{self.base_url}/the25/jvp.html", "category": "Cardiovascular"},
            
            # Respiratory
            {"title": "Pneumonia", "url": f"{self.base_url}/the25/pneumonia.html", "category": "Respiratory"},
            {"title": "COPD", "url": f"{self.base_url}/the25/copd.html", "category": "Respiratory"},
            
            # Neurological
            {"title": "Reflexes", "url": f"{self.base_url}/the25/reflexes.html", "category": "Neurological"},
            {"title": "Cerebellar Exam", "url": f"{self.base_url}/the25/cerebellar.html", "category": "Neurological"},
            {"title": "Gait", "url": f"{self.base_url}/the25/gait.html", "category": "Neurological"},
            
            # Musculoskeletal
            {"title": "Hand Examination", "url": f"{self.base_url}/the25/hand.html", "category": "Musculoskeletal"},
            {"title": "Hip Examination", "url": f"{self.base_url}/the25/hip.html", "category": "Musculoskeletal"},
            {"title": "Ankle and Foot", "url": f"{self.base_url}/the25/anklefoot.html", "category": "Musculoskeletal"},
            {"title": "Shoulder Examination", "url": f"{self.base_url}/the25/shoulder.html", "category": "Musculoskeletal"},
            {"title": "Knee Examination", "url": f"{self.base_url}/the25/knee.html", "category": "Musculoskeletal"},
            
            # Abdominal
            {"title": "Ascites", "url": f"{self.base_url}/the25/ascites.html", "category": "Abdominal"},
            {"title": "Liver Exam", "url": f"{self.base_url}/the25/liver.html", "category": "Abdominal"},
            {"title": "Spleen Exam", "url": f"{self.base_url}/the25/spleen.html", "category": "Abdominal"},
            
            # Dermatological
            {"title": "Acne", "url": f"{self.base_url}/the25/acne.html", "category": "Dermatological"},
            {"title": "Mole Examination", "url": f"{self.base_url}/the25/mole.html", "category": "Dermatological"},
            
            # Other
            {"title": "Breast Examination", "url": f"{self.base_url}/the25/breast.html", "category": "Other"},
            {"title": "Thyroid Examination", "url": f"{self.base_url}/the25/thyroid.html", "category": "HEENT"},
        ]
        
        return manual_pages


# Usage example
if __name__ == "__main__":
    # Initialize scraper
    scraper = StanfordMedicine25Scraper("StanfordPE")
    
    # Option 1: Automatic discovery and scraping
    results = scraper.scrape_all(delay=2.0)
    
    # Option 2: Scrape specific pages only
    # specific_pages = scraper._get_manual_page_list()
    # for page in specific_pages:
    #     scraped = scraper.scrape_page(page['url'])
    #     if scraped:
    #         scraper.save_content(page, scraped)