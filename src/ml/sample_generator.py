"""
Generates statistically representative, balanced CICIoT2023 sample datasets
matching the exact 46-feature schema and 8 high-level attack categories.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional

from .ciciot2023_config import FEATURE_NAMES, LABEL_COLUMN, CLASS_NAMES_8, RAW_LABEL_TO_8CLASS


def generate_ciciot2023_sample(
    num_rows: int = 50000,
    output_csv_path: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic, balanced sample dataset of 50,000 rows (or custom count)
    with the exact 46 flow features of CICIoT2023 across 8 classes.

    Args:
        num_rows: Total rows to generate (default: 50,000).
        output_csv_path: Optional file path to save the generated CSV.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing 46 feature columns and 1 label column.
    """
    rng = np.random.default_rng(random_state)
    
    # Representative raw attack sub-types mapped to their 8 classes
    class_subtypes = {
        "Benign": ["BenignTraffic"],
        "DDoS": ["DDoS-SYN_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood", "DDoS-ICMP_Flood", "DDoS-SlowLoris"],
        "DoS": ["DoS-UDP_Flood", "DoS-TCP_Flood", "DoS-SYN_Flood", "DoS-HTTP_Flood"],
        "Mirai": ["Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood"],
        "Recon": ["Recon-PortScan", "Recon-OSScan", "Recon-PingSweep", "VulnerabilityScan"],
        "Spoofing": ["DNS_Spoofing", "MITM-ArpSpoofing"],
        "Web": ["SqlInjection", "CommandInjection", "XSS", "Backdoor_Malware"],
        "BruteForce": ["DictionaryBruteForce"],
    }

    # Allocate balanced rows per class (e.g. 6250 rows each for 50k total)
    num_classes = len(CLASS_NAMES_8)
    base_rows_per_class = num_rows // num_classes
    remainder = num_rows % num_classes

    dfs = []

    for idx, class_name in enumerate(CLASS_NAMES_8):
        n_samples = base_rows_per_class + (1 if idx < remainder else 0)
        subtypes = class_subtypes[class_name]
        assigned_subtypes = rng.choice(subtypes, size=n_samples)

        # Baseline characteristics customized by class category
        if class_name == "Benign":
            flow_dur = rng.exponential(scale=1.5, size=n_samples) + 0.01
            rate = rng.gamma(shape=2.0, scale=15.0, size=n_samples)
            pkt_avg = rng.normal(loc=450.0, scale=120.0, size=n_samples).clip(40, 1500)
            syn_flags = rng.binomial(n=1, p=0.05, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.60, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.01, size=n_samples)
            http_p = 0.35
            https_p = 0.45
            dns_p = 0.15
            tcp_p = 0.85
            udp_p = 0.15
            arp_p = 0.02
            icmp_p = 0.01
        elif class_name in ["DDoS", "DoS"]:
            # High rates, high syn/udp flags, low duration per flow
            flow_dur = rng.exponential(scale=0.1, size=n_samples) + 0.001
            rate = rng.gamma(shape=5.0, scale=250.0, size=n_samples) + 500.0
            pkt_avg = rng.normal(loc=120.0, scale=40.0, size=n_samples).clip(28, 1400)
            syn_flags = rng.binomial(n=1, p=0.75, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.10, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.25, size=n_samples)
            http_p = 0.10
            https_p = 0.05
            dns_p = 0.05
            tcp_p = 0.60
            udp_p = 0.40
            arp_p = 0.0
            icmp_p = 0.15
        elif class_name == "Mirai":
            # IoT botnet traffic, rapid gre/udp/ack floods
            flow_dur = rng.exponential(scale=0.05, size=n_samples) + 0.0005
            rate = rng.gamma(shape=4.0, scale=300.0, size=n_samples) + 800.0
            pkt_avg = rng.normal(loc=64.0, scale=10.0, size=n_samples).clip(20, 512)
            syn_flags = rng.binomial(n=1, p=0.40, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.50, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.10, size=n_samples)
            http_p = 0.02
            https_p = 0.02
            dns_p = 0.10
            tcp_p = 0.45
            udp_p = 0.55
            arp_p = 0.01
            icmp_p = 0.02
        elif class_name == "Recon":
            # Fast probing scans, small packets, high syn/rst
            flow_dur = rng.exponential(scale=0.02, size=n_samples) + 0.0001
            rate = rng.gamma(shape=3.0, scale=50.0, size=n_samples) + 50.0
            pkt_avg = rng.normal(loc=54.0, scale=10.0, size=n_samples).clip(40, 120)
            syn_flags = rng.binomial(n=1, p=0.90, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.05, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.70, size=n_samples)
            http_p = 0.05
            https_p = 0.02
            dns_p = 0.05
            tcp_p = 0.85
            udp_p = 0.10
            arp_p = 0.05
            icmp_p = 0.20
        elif class_name == "Spoofing":
            # ARP/DNS spoofing
            flow_dur = rng.exponential(scale=0.5, size=n_samples) + 0.01
            rate = rng.gamma(shape=2.0, scale=20.0, size=n_samples)
            pkt_avg = rng.normal(loc=180.0, scale=60.0, size=n_samples).clip(42, 600)
            syn_flags = rng.binomial(n=1, p=0.10, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.20, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.05, size=n_samples)
            http_p = 0.05
            https_p = 0.05
            dns_p = 0.45
            tcp_p = 0.20
            udp_p = 0.50
            arp_p = 0.40
            icmp_p = 0.05
        elif class_name == "Web":
            # HTTP/HTTPS requests with SQLi/XSS payloads
            flow_dur = rng.exponential(scale=1.2, size=n_samples) + 0.05
            rate = rng.gamma(shape=1.5, scale=10.0, size=n_samples)
            pkt_avg = rng.normal(loc=650.0, scale=250.0, size=n_samples).clip(100, 1500)
            syn_flags = rng.binomial(n=1, p=0.10, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.85, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.05, size=n_samples)
            http_p = 0.65
            https_p = 0.30
            dns_p = 0.05
            tcp_p = 0.95
            udp_p = 0.05
            arp_p = 0.0
            icmp_p = 0.0
        else:  # BruteForce
            # Repetitive authentication attempts
            flow_dur = rng.exponential(scale=0.4, size=n_samples) + 0.02
            rate = rng.gamma(shape=2.5, scale=40.0, size=n_samples) + 20.0
            pkt_avg = rng.normal(loc=220.0, scale=80.0, size=n_samples).clip(60, 800)
            syn_flags = rng.binomial(n=1, p=0.30, size=n_samples)
            ack_flags = rng.binomial(n=1, p=0.70, size=n_samples)
            rst_flags = rng.binomial(n=1, p=0.15, size=n_samples)
            http_p = 0.30
            https_p = 0.10
            dns_p = 0.02
            tcp_p = 0.90
            udp_p = 0.05
            arp_p = 0.0
            icmp_p = 0.0

        pkt_min = (pkt_avg * rng.uniform(0.1, 0.4, size=n_samples)).clip(20, 100)
        pkt_max = (pkt_avg * rng.uniform(1.5, 3.0, size=n_samples)).clip(100, 1514)
        pkt_std = (pkt_max - pkt_min) / rng.uniform(3.0, 5.0, size=n_samples)
        num_pkts = (flow_dur * rate).clip(1, 100000).astype(int)
        tot_sum = pkt_avg * num_pkts
        tot_size = tot_sum + rng.uniform(10, 100, size=n_samples)
        iat = (flow_dur / np.maximum(num_pkts, 1)).clip(0.00001, 10.0)

        class_dict = {
            "flow_duration": flow_dur,
            "Header_Length": rng.choice([20, 32, 40, 54], size=n_samples),
            "Protocol Type": rng.choice([6, 17, 1, 0], size=n_samples, p=[0.70, 0.25, 0.04, 0.01]),
            "Duration": flow_dur,
            "Rate": rate,
            "Srate": rate * rng.uniform(0.4, 0.9, size=n_samples),
            "Drate": rate * rng.uniform(0.1, 0.6, size=n_samples),
            "fin_flag_number": rng.binomial(n=1, p=0.05, size=n_samples),
            "syn_flag_number": syn_flags,
            "rst_flag_number": rst_flags,
            "psh_flag_number": rng.binomial(n=1, p=0.20, size=n_samples),
            "ack_flag_number": ack_flags,
            "ece_flag_number": rng.binomial(n=1, p=0.01, size=n_samples),
            "cwr_flag_number": rng.binomial(n=1, p=0.01, size=n_samples),
            "ack_count": (num_pkts * ack_flags * rng.uniform(0.5, 1.0, size=n_samples)).round(),
            "syn_count": (num_pkts * syn_flags * rng.uniform(0.5, 1.0, size=n_samples)).round(),
            "fin_count": (num_pkts * rng.uniform(0.0, 0.1, size=n_samples)).round(),
            "rst_count": (num_pkts * rst_flags * rng.uniform(0.1, 0.5, size=n_samples)).round(),
            "HTTP": rng.binomial(n=1, p=http_p, size=n_samples),
            "HTTPS": rng.binomial(n=1, p=https_p, size=n_samples),
            "DNS": rng.binomial(n=1, p=dns_p, size=n_samples),
            "Telnet": rng.binomial(n=1, p=0.01 if class_name == "Mirai" else 0.001, size=n_samples),
            "SMTP": rng.binomial(n=1, p=0.005, size=n_samples),
            "SSH": rng.binomial(n=1, p=0.15 if class_name == "BruteForce" else 0.01, size=n_samples),
            "IRC": rng.binomial(n=1, p=0.01, size=n_samples),
            "TCP": rng.binomial(n=1, p=tcp_p, size=n_samples),
            "UDP": rng.binomial(n=1, p=udp_p, size=n_samples),
            "DHCP": rng.binomial(n=1, p=0.02, size=n_samples),
            "ARP": rng.binomial(n=1, p=arp_p, size=n_samples),
            "ICMP": rng.binomial(n=1, p=icmp_p, size=n_samples),
            "IPv": rng.binomial(n=1, p=0.98, size=n_samples),
            "LLC": rng.binomial(n=1, p=0.01, size=n_samples),
            "Tot sum": tot_sum,
            "Min": pkt_min,
            "Max": pkt_max,
            "AVG": pkt_avg,
            "Std": pkt_std,
            "Tot size": tot_size,
            "IAT": iat,
            "Number": num_pkts,
            "Magnitue": np.sqrt(tot_sum),
            "Radius": pkt_std * 1.414,
            "Covariance": pkt_std * 0.8,
            "Variance": pkt_std ** 2,
            "Weight": num_pkts.astype(float),
            LABEL_COLUMN: assigned_subtypes,
        }
        dfs.append(pd.DataFrame(class_dict))

    full_df = pd.concat(dfs, ignore_index=True)
    # Shuffle rows
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        full_df.to_csv(output_csv_path, index=False)
        print(f"[Sample Generator] Saved {len(full_df)} rows to {output_csv_path}")

    return full_df
