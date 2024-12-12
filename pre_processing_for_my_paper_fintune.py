from scapy.all import *
import time
import csv
import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from threading import Lock
from scapy.compat import raw
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scapy.layers.dns import DNS
from scapy.utils import PcapReader

from threading import Lock
mutex = Lock()
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
        # print('----------------------')
    # if TCP in packet:
    #     packet[TCP].src = "0"
    #     packet[]
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

def files_handle(root,k):
    global count
    files_list = os.listdir(root)
    count = 0
    # with ThreadPoolExecutor(max_workers=8) as executor:
    for file in tqdm.tqdm(files_list):
        if count >= flag:
            return

        file_path = os.path.join(root,file)
        pcap_list = os.listdir(file_path)
        for pcap in pcap_list:
            pcap_path = os.path.join(file_path,pcap)
            try:
                extraBytes(pcap_path,k)

            except:
                pass
            # executor.submit(extraBytes, file_path)

def extraBytes(pcapPath,k):
    global count
    pkl_origin = PcapReader(pcapPath)
    pcap_name = os.path.basename(pcapPath)
    app_id = pcap_name.split('.')[0].split('-')[0]
    if int(app_id) not in app_dict.keys():
        return

    for pk in pkl_origin:
        try:
            if should_omit_packet(pk):
                continue
            pk = remove_ether_header(pk)
            pk = pad_udp(pk)
            pk = mask_ip(pk)
            if len(raw(pk).hex()) <= 400:
                continue
            Bytes = raw(pk).hex()[0:400]
            bytes_str = ''
            for i in range(0,len(Bytes),2):
                doubleByte = Bytes[i:i+2]
                bytes_str += doubleByte + ' '
            bytes_str = bytes_str[0:-1]
            bytes_str += '\t'+str(k) + '\t'
            bytes_str += app_id
            bytes_str += '\n'
            mutex.acquire()
            with open('fine_tuning_data.txt','a+') as file:
                file.write(bytes_str)
                count += 1
            mutex.release()
        except:
            continue

app_dict = {1:'google.com',2:'youtube.com',3:'facebook.com',4:'baidu.com',
            6:'instagram.com',7:'yahoo.com',10:'amazon.com'
            ,12:'twitter.com',15:'microsoft.com',20:'bing.com',21:'reddit.com',
            31:'apple.com',22:'netflix.com',33:'tiktok.com',17:'github.com',41:'twitch.tv',50:'telegram.org',
            61:'quora.com',9:'wikipedia.org',26:'zoom.us',19:'taobao.com',16:'csdn.net',62:'douban.com',69:'deepl.com'
            }
bridge = {1:'Non-Tor',2:'Norm_Tor',3:'SNOWFLAKE',4:'OBFS4_0',5:'OBFS4_1',6:'OBFS4_2'}

def main():
    root = 'E:\maxFlow'
    for k in bridge.keys():
        root_path = os.path.join(root,bridge[k])
        print(root_path)
        files_handle(root_path, k)
if __name__ == '__main__':
    flag = 50000
    count = 0
    main()