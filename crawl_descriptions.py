import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Đọc CSV để lấy danh sách loài rắn
csv_path = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\archive\Csv\train.csv"
df = pd.read_csv(csv_path)

# Lấy tất cả unique binomials
unique_binomials = df['binomial'].unique()
print(f"Processing {len(unique_binomials)} species")

# Hàm để lấy mô tả từ Wikipedia
def get_wiki_description(binomial, session, max_retries=3):
    # Chuyển binomial thành URL format (thay space bằng underscore)
    species_name = binomial.replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/{species_name}"

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Tìm phần mô tả chính (lead section): paragraph đầu tiên sau infobox
            content = soup.find('div', {'class': 'mw-parser-output'})
            if content:
                paragraphs = content.find_all('p', limit=3)  # Lấy 2-3 đoạn đầu
                description = ''
                for p in paragraphs:
                    text = p.get_text()
                    # Loại bỏ citations [1], [2], etc.
                    text = re.sub(r'\[.*?\]', '', text)
                    if text.strip() and len(text.strip()) > 20:
                        description += text.strip() + ' '
                return description.strip() if description.strip() else "No description found"
            else:
                return "Page found but no content"

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"  Timeout, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2 ** attempt)
            else:
                return "Error: Timeout"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"  Connection error, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2 ** attempt)
            else:
                return "Error: Connection failed"
        except requests.RequestException as e:
            return f"Error: {str(e)[:50]}"
        except Exception as e:
            return f"Error: {str(e)[:50]}"

    return "Error: Max retries exceeded"

# Thiết lập session với retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

# Thu thập mô tả
descriptions = {}
success_count = 0
error_count = 0

for idx, binomial in enumerate(unique_binomials, 1):
    print(f"[{idx}/{len(unique_binomials)}] Fetching description for {binomial}...")
    desc = get_wiki_description(binomial, session, max_retries=3)
    descriptions[binomial] = desc
    
    if not desc.startswith("Error"):
        success_count += 1
    else:
        error_count += 1
    
    time.sleep(0.5)  # Delay để tránh bị block

# Tạo DataFrame và lưu
desc_df = pd.DataFrame(list(descriptions.items()), columns=['binomial', 'description'])
output_path = r"d:\Dai Hoc\CUOI\XULYANH\TimKiemRan\wiki_descriptions.csv"
desc_df.to_csv(output_path, index=False)
print(f"\nDescriptions saved to {output_path}")
print(f"Success: {success_count}/{len(unique_binomials)}, Errors: {error_count}")

# Hiển thị mẫu
print("\nSample descriptions:")
for binomial, desc in list(descriptions.items())[:5]:
    print(f"\n{binomial}:\n{desc[:300]}...")