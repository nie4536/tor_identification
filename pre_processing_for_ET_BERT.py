import numpy as np
import gzip
import json
from scapy.compat import raw
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scapy.layers.dns import DNS
from scapy.utils import PcapReader
import binascii
import csv,os

def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False

def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"
        packet[IP].proto = 0
        # print(packet[IP].proto)
    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet
def packet_to_sparse_array(packet, max_length=1500):

    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    # print(arr)
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr_list = list(arr)
    return arr_list

def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)
    return arr

app_dict = {1:'google.com',2:'youtube.com',3:'facebook.com',4:'baidu.com',
            6:'instagram.com',7:'yahoo.com',10:'amazon.com'
            ,12:'twitter.com',15:'microsoft.com',20:'bing.com',21:'reddit.com',
            31:'apple.com',22:'netflix.com',33:'tiktok.com',17:'github.com',41:'twitch.tv',50:'telegram.org',
            61:'quora.com',9:'wikipedia.org',26:'zoom.us',19:'taobao.com',16:'csdn.net',62:'douban.com',69:'deepl.com'
            }
bridge = {1:'Non-Tor',2:'Norm_Tor',3:'SNOWFLAKE',4:'OBFS4_0',5:'OBFS4_1',6:'OBFS4_2'}
def file_handle(root,k):
    file = open(r'ET-BERT_data.txt',mode='a+',encoding='utf-8',newline='')
    swriter = csv.writer(file)
    dirs = os.listdir(root)
    pcap_path_list = [os.path.join(root, dir) for dir in dirs]
    p_num = 0
    for path in pcap_path_list:
        for pcap_file in os.listdir(path):
            pcap_final_path = os.path.join(path,pcap_file)
            # app_id = pcap_file.split('.')[0].split('-')[0]
            # if int(app_id) not in app_dict.keys():
            #     continue

            for p in PcapReader(pcap_final_path):
                if p_num >= 50000:
                    return
                row = []
                if UDP not in p:
                    continue
                arr = transform_packet(p)
                if arr is None:
                    continue
                row.append(k)
                # row.append(int(app_id))
                row.append(0)
                row.extend(arr)
                swriter.writerow(row)
                p_num+=1

def main():
    root = 'E:\maxFlow'
    for k in bridge.keys():
        root_path = os.path.join(root,bridge[k])
        print(root_path)
        file_handle(root_path, k)
if __name__ == '__main__':
    file_handle(r'E:\maxflow',0)
    # main()
