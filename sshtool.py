import paramiko as ssh
from scp import SCPClient

import logging

class SSHTool():
    def __init__(self, host, user, auth,
                 via=None, via_user=None, via_auth=None):
        
        if via:
            t0 = ssh.Transport(via)
            t0.start_client()
            t0.auth_password(via_user, via_auth)
            # setup forwarding from 127.0.0.1:<free_random_port> to |host|
            channel = t0.open_channel('direct-tcpip', host, ('127.0.0.1', 0))
            self.transport = ssh.Transport(channel)
        else:
            self.transport = ssh.Transport(host)
        self.transport.start_client()
        self.transport.auth_password(user, auth)

    def run(self, cmd):
        ch = self.transport.open_session()
        ch.set_combine_stderr(True)
        ch.exec_command(cmd)
        retcode = ch.recv_exit_status()
        logging.info('cmd finished w/ retcode %d', retcode)
        buf = ''
        while ch.recv_ready():
            buf += ch.recv(1024).decode("utf-8")
        return (buf, retcode)

    def transfer(self, remote_path, local_path):
        scp = SCPClient(self.transport)
        scp.get(remote_path, local_path)