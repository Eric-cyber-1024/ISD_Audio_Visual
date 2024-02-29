import paramiko

# Set the remote server details
hostname = '192.168.70.52'
username = 'pi'
password = 'pi'
command = 'cd ~/workspace/python ; python test_xy_uart.py'  # The command you want to run remotely

# Create an SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Connect to the remote server
    client.connect(hostname, username=username, password=password)

    # Run the remote command
    sPortName='/dev/ttyACM0'
    sArg='0'
    stdin, stdout, stderr = client.exec_command('%s %s %s' %(command,sPortName,sArg))

    # Read the output of the command
    output = stdout.read().decode()

    # Print the output
    print(output)

finally:
    # Close the SSH connection
    client.close()