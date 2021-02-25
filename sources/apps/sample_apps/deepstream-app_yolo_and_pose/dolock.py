#!/usr/bin/python
import sys
import os
import select
from  socket import *
import binascii

HOST = ''
PORT = 21567
BUFSIZE = 1024

server_fd = socket(AF_INET, SOCK_DGRAM)
server_fd.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)

server_fd.bind((HOST, PORT))

inputs = [server_fd]
client_address = None
conn = None

if len(sys.argv) == 2:
        if sys.argv[1] == "-h":
            print("Usage: python control.py 10.193.205.84")
            sys.exit()
        HOST_SERVER=sys.argv[1]
else:
        HOST_SERVER = '<broadcast>'
print("use addr:"+ HOST_SERVER)

PORT_SERVER=PORT
ADDR = (HOST_SERVER, PORT_SERVER)

msg = "LOCK"
msg_as_bytes = str.encode(msg)
server_fd.sendto(msg_as_bytes,ADDR)
 

count=2
flag=0
while count>0:
   count=count-1
   readable = select.select(inputs, [], [], 1.0)[0]
   for s in readable:
       data, client_address = s.recvfrom(1024)
       if data != msg:
        #print(s)
        #print(client_address)
        print(data)
        flag = 1
        break
   if flag == 0:
      if count ==0:
           print("Time out.")
      else:
           break

server_fd.close()
