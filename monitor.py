#!/usr/bin/env python3
"""
Monitor de population pour serveurs Source/GMod.
Interroge chaque serveur via le protocole A2S_INFO (aucune dependance externe,
uniquement des requetes UDP sortantes -> fonctionne en local sans probleme).

Usage:
    python3 monitor.py            # tourne en continu, interroge toutes les X minutes
    python3 monitor.py --once     # une seule requete puis quitte (pratique pour tester / cron)
"""

import socket
import struct
import time
import csv
import os
import argparse
from datetime import datetime

# ------------------------------------------------------------------
# Configuration des serveurs a suivre : nom -> (ip, port)
# ------------------------------------------------------------------
SERVERS = {
    "AETHER":       ("185.55.240.135", 27017),
    "AETHER_EVENT": ("185.55.240.135", 27015),
    "NEXUS":        ("31.56.58.42", 27015),
    "AXIOM":        ("194.31.143.133", 27032),
    "COSMOS":       ("185.29.166.154", 27015),
}

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.csv")
INTERVAL_SECONDS = 10 * 60  # 10 minutes entre chaque relevé
SOCKET_TIMEOUT = 3.0

A2S_INFO_REQUEST = b"\xFF\xFF\xFF\xFF\x54Source Engine Query\x00"


def read_cstring(data: bytes, offset: int):
    """Lit une chaine terminee par un octet nul a partir de offset."""
    end = data.index(b"\x00", offset)
    return data[offset:end].decode("utf-8", errors="replace"), end + 1


def query_server(ip: str, port: int):
    """
    Envoie une requete A2S_INFO et parse la reponse.
    Retourne un dict {name, map, players, max_players, bots} ou None si injoignable.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.sendto(A2S_INFO_REQUEST, (ip, port))
        data, _ = sock.recvfrom(4096)
    except (socket.timeout, OSError):
        return None
    finally:
        sock.close()

    # Certains serveurs renvoient un challenge (0x41) au lieu de la reponse directe.
    # On relance alors la requete en incluant le challenge recu.
    if len(data) >= 5 and data[4] == 0x41:
        challenge = data[5:9]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(SOCKET_TIMEOUT)
        try:
            sock.sendto(A2S_INFO_REQUEST + challenge, (ip, port))
            data, _ = sock.recvfrom(4096)
        except (socket.timeout, OSError):
            return None
        finally:
            sock.close()

    if len(data) < 6 or data[4] != 0x49:  # 'I'
        return None

    offset = 5
    offset += 1  # protocol version (byte)
    name, offset = read_cstring(data, offset)
    map_name, offset = read_cstring(data, offset)
    _folder, offset = read_cstring(data, offset)
    _game, offset = read_cstring(data, offset)

    offset += 2  # ID (short)
    players = data[offset]; offset += 1
    max_players = data[offset]; offset += 1
    bots = data[offset]; offset += 1

    return {
        "name": name,
        "map": map_name,
        "players": players,
        "max_players": max_players,
        "bots": bots,
    }


def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "server_label", "players", "max_players", "bots", "map", "online"])


def poll_once():
    ensure_csv_header()
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for label, (ip, port) in SERVERS.items():
        info = query_server(ip, port)
        if info is None:
            rows.append([now, label, "", "", "", "", 0])
            print(f"[{now}] {label:15s} -> injoignable")
        else:
            rows.append([now, label, info["players"], info["max_players"], info["bots"], info["map"], 1])
            print(f"[{now}] {label:15s} -> {info['players']}/{info['max_players']} joueurs (carte: {info['map']})")

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Fait une seule serie de requetes puis quitte")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECONDS, help="Intervalle en secondes entre les relèves")
    args = parser.parse_args()

    if args.once:
        poll_once()
        return

    print(f"Monitoring demarre. Relevé toutes les {args.interval // 60} minutes. Ctrl+C pour arreter.")
    try:
        while True:
            poll_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nArret du monitoring.")


if __name__ == "__main__":
    main()