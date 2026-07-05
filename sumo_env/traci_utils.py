"""TraCI startup helpers."""

import socket

import traci


def free_tcp_port() -> int:
    """Ask the OS for an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_traci(cmd, **kwargs):
    """Start TraCI with an explicit port for SUMO builds that need one."""
    kwargs.setdefault("port", free_tcp_port())
    return traci.start(cmd, **kwargs)
