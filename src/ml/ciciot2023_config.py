"""
Configuration and constants for CICIoT2023 dataset processing.
Defines 46 statistical flow features, 8-class categories, binary mappings, and URLs.
"""

# The 46 standard numerical flow features in CICIoT2023
FEATURE_NAMES = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]

LABEL_COLUMN = "label"

# Hierarchical mapping from specific CICIoT2023 raw labels (34 types) to 8 high-level classes
RAW_LABEL_TO_8CLASS = {
    # Benign
    "BenignTraffic": "Benign",
    "Benign": "Benign",
    
    # DDoS
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    
    # DoS
    "DoS-UDP_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    
    # Mirai / Botnet
    "Mirai-greeth_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-ackflood": "Mirai",
    
    # Reconnaissance / Scanning
    "Recon-PingSweep": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "VulnerabilityScan": "Recon",
    
    # Spoofing / MITM
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    
    # Web / Injection / Malware
    "BrowserHijacking": "Web",
    "Backdoor_Malware": "Web",
    "XSS": "Web",
    "SqlInjection": "Web",
    "CommandInjection": "Web",
    "Uploading_Attack": "Web",
    
    # Brute Force
    "DictionaryBruteForce": "BruteForce",
}

CLASS_NAMES_8 = [
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
    "BruteForce",
]

CATEGORY_MAPPING_8CLASS = RAW_LABEL_TO_8CLASS

# Base URL for CIC repository sample archives
CIC_DATASET_BASE_URL = "http://205.174.165.80/CICDataset/CICIoT2023/Dataset/CSVs/"
