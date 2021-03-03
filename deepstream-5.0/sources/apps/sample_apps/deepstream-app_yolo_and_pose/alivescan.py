#!/usr/bin/python
from socket import *
import os
import sys
import time

if len(sys.argv) == 2:
	if sys.argv[1] == "-h":
	    print "Usage: python AliveScan.py xx.xx.xx.xx"
	    sys.exit()
        HOST=sys.argv[1]
else:
	HOST = '<broadcast>'
print "use addr:"+ HOST

 
PORT = 21567

BUFSIZE = 1024
 
ADDR = (HOST, PORT)
 
udpCliSock = socket(AF_INET, SOCK_DGRAM)
#udpCliSock.bind(('', PORT))
udpCliSock.bind(('', 0))
udpCliSock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)


try:
    pid = os.fork()
    if pid == 0:
	while True:
	    data="ONLINE"
    	    print "Scan... "
    	    udpCliSock.sendto(data,ADDR)
	    time.sleep(5)
    else:
	while True:
    	    data1,ADDR1 = udpCliSock.recvfrom(BUFSIZE)
    	    if not data1:
                break
		
    	    print data1 
except OSError, e:
    pass
	
 
udpCliSock.close()
