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
from scapy.utils import PcapReader,PcapWriter

from threading import Lock
mutex = Lock()

#是否省略数据包
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


#剥离以太网帧的头和尾
def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet

#将报文中的源IP地址和目的IP地址替换为“0.0.0.0”，源端口号和目的端口号替换为0。
def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"
        packet[IP].sport = 0
        # print(packet[IP].proto)
        # print('----------------------')
    # if TCP in packet:
    #     packet[TCP].src = "0"
    #     packet[]
    return packet

#TCP和UDP的报头长度不同，在UDP报头后填充12字节的0x00，以保证输入到模型的数据长度一致。
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

def files_handle(root):
    files_list = os.listdir(root)
    count = 0
    # with ThreadPoolExecutor(max_workers=8) as executor:
    for file in tqdm.tqdm(files_list):
        # if count >= flag:
        #     return

        file_path = os.path.join(root,file)
        pcap_list = os.listdir(file_path)
        for pcap in pcap_list:
            pcap_path = os.path.join(file_path,pcap)
            try:
                extraBytes(pcap_path)
                count += 1
            except:
                pass

def extraBytes(pcapPath):
    pkl_origin = PcapReader(pcapPath)
    webID = (pcapPath).split(".")[0]
    wirters=PcapWriter("{}.pcap".format(webID))
    i=0
    for pk in pkl_origin:
        #try:
        if should_omit_packet(pk):
            continue
        else:
            pk = remove_ether_header(pk)
            pk = pad_udp(pk)
            pk = mask_ip(pk)
            i+=1
            print(i)
        wirters.write(pkt=pk)
            # if len(raw(pk).hex()) <= 600:
            #     continue
            # Bytes = raw(pk).hex()[0:600]
            # bytes_str = ''
            # for i in range(0,len(Bytes),2):
            #     doubleByte = Bytes[i:i+2]
            #     bytes_str += doubleByte + ' '
            # bytes_str = bytes_str[0:-1]
            # bytes_str += '\n'
            # mutex.acquire()
            # with open('pretext.txt','a+') as file:
            #     file.write(bytes_str)
            # mutex.release()
        # except:
        #    continue

if __name__ == '__main__':
    flag = 50000
    files_handle(r'E:\maxFlow')
    #extraBytes(r'E:\maxFlow/facebook_chat_4b/facebook_chat_4b.pcap.TCP_131-202-240-242_32576_173-252-100-27_443.pcap')
