import re
from urllib.parse import urlparse

import pandas as pd
import tldextract

# uses the suffix list that ships with the package, so no internet is needed
split_domain = tldextract.TLDExtract(suffix_list_urls=())

KEYWORDS = ["login", "signin", "verify", "secure", "account", "update",
            "bank", "confirm", "password", "billing", "paypal"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"]
RISKY_TLDS = ["xyz", "tk", "ml", "ga", "cf", "gq", "top", "icu", "click"]


def get_features(url):
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url

    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = urlparse("http://invalid")

    # only the domain part is analysed; "www." is dropped so it can't act as a shortcut
    host = (parsed.hostname or "").lower()
    host = re.sub(r"^www\.", "", host)
    parts = split_domain(host)

    subdomains = [s for s in parts.subdomain.split(".") if s]
    digits = sum(c.isdigit() for c in host)

    return {
        "hostname_length": len(host),
        "domain_length": len(parts.domain),
        "subdomain_count": len(subdomains),
        "tld_length": len(parts.suffix),
        "dot_count": host.count("."),
        "hyphen_count": host.count("-"),
        "domain_hyphen_count": parts.domain.count("-"),
        "digit_count": digits,
        "digit_ratio": digits / max(len(host), 1),
        "has_at": int("@" in parsed.netloc),
        "has_https": int(parsed.scheme == "https"),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "risky_tld": int(parts.suffix.split(".")[-1] in RISKY_TLDS),
        "is_shortener": int(f"{parts.domain}.{parts.suffix}" in SHORTENERS),
        "keyword_count": sum(k in host for k in KEYWORDS),
    }


FEATURE_COLUMNS = list(get_features("example.com").keys())


def build_feature_table(urls):
    rows = [get_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://secure-login.paypal.com.verify-id.xyz/update?id=123",
        "http://192.168.1.1/login.php",
    ]
    print(build_feature_table(test_urls).T)