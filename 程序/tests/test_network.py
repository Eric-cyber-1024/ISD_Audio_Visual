'''

simple script to test networking stuffs

'''
import netifaces
import socket
import time

class networkController:
    def __init__(self):
        pass


    def tryPing(self,host,count=4, toPrint=False):
        '''
        ty to ping a host using ICMP socket

        
        Args:
            host: the host to ping with
            count: the number of ping trials (default is 4)
            toPrint: print out debug messages if True (default is False)

        Returns:
            True if able to ping the host
            results TTL, RTT


        Raises:
            None

        Example usage:
            

        '''
        
        # Get the IP address of the desired network interface
        interface = netifaces.gateways()['default'][netifaces.AF_INET][1]
        ip_address = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']

        # Create a socket for ICMP requests
        icmp_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        icmp_socket.bind((ip_address, 0))

        # Set the timeout for receiving ICMP reply
        icmp_socket.settimeout(1)

        failedCount=0
        results=[]
        for sequence_number in range(1, count + 1):
            # Build an ICMP echo request packet
            packet_id = 1234
            packet = bytearray([8, 0, 0, 0, packet_id // 256, packet_id % 256, sequence_number // 256, sequence_number % 256])

            # Calculate the ICMP checksum
            checksum = 0
            for i in range(0, len(packet), 2):
                checksum += packet[i] * 256 + packet[i + 1]
            checksum = (checksum & 0xffff) + (checksum >> 16)
            checksum = (~checksum & 0xffff)
            packet[2] = checksum // 256
            packet[3] = checksum % 256

            try:
                # Record the start time
                start_time = time.time()

                # Send the ICMP echo request
                icmp_socket.sendto(packet, (host, 0))

                # Receive the ICMP reply
                reply, address = icmp_socket.recvfrom(1024)

                # Extract the TTL value from the IP header
                ttl = reply[8]

                # Calculate the round-trip time
                rtt = (time.time() - start_time) * 1000  # in milliseconds

                if toPrint:
                    # Print the reply, TTL value, and round-trip time
                    print("Ping reply received from:", address[0])
                    print("TTL:", ttl)
                    print("RTT:", round(rtt, 2), "ms")

                results.append([ttl,rtt])
            except socket.timeout:
                # Handle timeout if no reply received
                if toPrint:
                    print("Ping request timed out")
                
                # increment failedCount
                failedCount+=1

        # Close the ICMP socket
        icmp_socket.close()


        if failedCount>1:
            return False,results
        else:
            return True,results


    def list_network_adapters(self):
        adapters = netifaces.interfaces()
        for adapter in adapters:
            addresses = netifaces.ifaddresses(adapter)
            if netifaces.AF_INET in addresses:
                ipv4_addresses = [addr['addr'] for addr in addresses[netifaces.AF_INET]]
            else:
                ipv4_addresses = []

            if netifaces.AF_INET6 in addresses:
                ipv6_addresses = [addr['addr'] for addr in addresses[netifaces.AF_INET6]]
            else:
                ipv6_addresses = []

            print(f"Adapter: {adapter}")
            print(f"IPv4 addresses: {ipv4_addresses}")
            print(f"IPv6 addresses: {ipv6_addresses}")
            print()


if __name__ == '__main__':
    myNetworkController = networkController()
    myNetworkController.list_network_adapters()
    print(myNetworkController.tryPing('yahoo.com'))

    # step 1: make sure the ethernet adapter is there
    # step 2: try to ping 192.168.1.40 to see if it's there

    