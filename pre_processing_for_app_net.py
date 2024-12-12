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

#是否省略数据包
def should_omit_packet(packet):
    # 设置了 SYN、ACK 或 FIN 标志以及空有效负载或仅填充有效负载的 TCP 数据包。
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS 段
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
def packet_to_sparse_array(packet, max_length=1024):
    # print(packet.original)

    Bytes = raw(packet).hex()
    # print(Bytes)

    return Bytes

def transform_packet(packet):
    if should_omit_packet(packet):
        return None
    # print((packet.original.hex()))
    packet = remove_ether_header(packet)
    # print((packet.original.hex()))
    packet = pad_udp(packet)
    # print((raw(packet).hex()))
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
    b_num =0
    l_num = 0
    file = open(r'app_net_length.csv',mode='a+',encoding='utf-8',newline='')
    swriter = csv.writer(file)
    dirs = os.listdir(root)
    pcap_path_list = [os.path.join(root, dir) for dir in dirs]
    p_num = 0
    for path in pcap_path_list:
        for pcap_file in os.listdir(path):
            if b_num!=l_num:
                print('!!!!!!!!!!!!!!!!!')
            pcap_final_path = os.path.join(path,pcap_file)
            app_id = pcap_file.split('.')[0].split('-')[0]
            if int(app_id) not in app_dict.keys():
                continue
            # print(pcap_final_path)
            bytes=''
            for p in PcapReader(pcap_final_path):
                arr = transform_packet(p)
                if arr is None:
                    continue
                bytes+=arr
                if len(bytes) < 2048:
                    continue
                bytes = bytes[0:2048]
                break
            row = []
            for p in PcapReader(pcap_final_path):
                if len(row) >= 20:
                    break
                if should_omit_packet(p):
                    continue
                p = remove_ether_header(p)
                p = pad_udp(p)
                p = mask_ip(p)
                l = len(p.payload)
                row.append(l)
            if len(row)<20:
                continue
            row.append(k)
            row.append(int(app_id))
            swriter.writerow(row)
            bytes_str = ''
            for i in range(0, len(bytes), 2):
                doubleByte = bytes[i:i + 2]
                bytes_str += doubleByte + ' '

            bytes_str = bytes_str[0:-1]
            bytes_str += '\t' + str(k) + '\t' + app_id + '\n'
            with open('app_net_bytes.txt', 'a+') as file:
                # print(arr)
                file.write(bytes_str)
                b_num += 1
            l_num += 1

                # p_num+=1
def main():
    root = 'E:\maxFlow'
    for k in bridge.keys():
        root_path = os.path.join(root,bridge[k])

        file_handle(root_path, k)
if __name__ == '__main__':
    file_handle(r'E:\maxFlow',2)
    # main()