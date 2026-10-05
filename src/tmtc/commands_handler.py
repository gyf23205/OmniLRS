__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"
import time
import omni.kit.app

import socket
import time
import threading

from src.tmtc.mdb_parsing_service import MdbParsingService

class CommandsHandler():
    UDP_RECV_MAX        = 4096  
    SOCKET_TIMEOUT_SEC  = 2.0 
    HEARTBEAT_EVERY_SEC = 10.0  # how often to log "waiting..." when idle

    def __init__(self, yamcs_processor, yamcs_instance_conf, mdb_files:list[str]):
        self._yamcs_processor = yamcs_processor
        self._commands_catalogue = {}
        self._registry = MdbParsingService.load_mdb_registry(mdb_files) 
        self._config_tc_socket(yamcs_instance_conf)
        self._start_tc_listener()


    def _config_tc_socket(self, yamcs_instance_conf):
        address = yamcs_instance_conf["tc_receive_address"]
        port = yamcs_instance_conf["tc_receive_port"]

        self._tc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._tc_socket.bind((address, port))
        except OSError as exc:
            # Almost always an earlier simulator still holding the port - including one merely
            # suspended with Ctrl-Z, which keeps its sockets. Say so here: the bare OSError surfaces
            # 40 lines deep in a Kit crash dump, where it reads like a fault in the rover code.
            # NOT fixed with SO_REUSEADDR on purpose: two processes bound to one telecommand port
            # would split the uplink between them, which is far worse than refusing to start.
            raise OSError(
                f"Cannot bind the telecommand port {address}:{port} ({exc.strerror}).\n"
                f"  Another simulator is probably still running or suspended. Find it with:\n"
                f"    ps -eo pid,stat,cmd | grep run_perseverance\n"
                f"  A 'T' in the STAT column means stopped, not dead - it still owns the port.\n"
                f"  Then: kill -9 <pid>"
            ) from exc

        self._tc_socket.settimeout(self.SOCKET_TIMEOUT_SEC)
        print("UDP bound to:", self._tc_socket.getsockname())

    def _start_tc_listener(self):
        self._tc_stop_event = threading.Event()
        self._tc_thread = threading.Thread(
            target=self._tc_listener_loop,
            name="tc-listener",
            daemon=True,
        )
        self._tc_thread.start()

    def _tc_listener_loop(self):
        last_heartbeat = 0.0
        while not self._tc_stop_event.is_set():  # while True:
            try:
                tc_data, addr = self._tc_socket.recvfrom(self.UDP_RECV_MAX)
            except socket.timeout:
                now = time.time()
                if now - last_heartbeat >= self.HEARTBEAT_EVERY_SEC:
                    print("Heartbeat: waiting for TC on", self._tc_socket.getsockname())
                    last_heartbeat = now
                continue

            decoded = MdbParsingService.decode_tc_payload(tc_data, self._registry)
            if decoded is None:
                # Skip the packet, do not leave the loop. Returning here killed the listener thread
                # on the first unrecognised payload, which silently deafened the rover to every
                # later command - and the usual cause is a stale mdb on one side of the link, so the
                # symptom looked nothing like the cause.
                print(f"Undecodable TC payload from {addr}: {tc_data[:32]!r}")
                continue

            command = self._commands_catalogue.get(decoded["full_name"])
            if command is None:
                print(f"Command '{decoded['full_name']}' not found in catalogue")
            else:
                self._execute(command, decoded["arguments"])
    
    def _execute(self, command, received_arguments):
        arg_names = command["args"]
        func = command["func"]

        if arg_names == []:
            func()
        else:
            args = [received_arguments[name] for name in arg_names]
            func(*args) 

    def add_command(self, command_name:str, func, args:list=[]):
        # Can be called from inside _init_commands_catalogue or externally.
        if command_name in self._commands_catalogue:
            raise Exception("Command named", str(command_name), "already exists")
        elif command_name == "":
            raise Exception("Command name can not be an empty string.")

        self._commands_catalogue[command_name] = {"func":func, "args":args}