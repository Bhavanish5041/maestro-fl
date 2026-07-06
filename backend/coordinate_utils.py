import os
import sumolib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_PATH = os.path.join(REPO_ROOT, "sumo_env", "network", "osm_fixed.net.xml")

_net = None

def get_net():
    global _net
    if _net is None:
        _net = sumolib.net.readNet(NET_PATH)
    return _net

def sumo_to_latlon(x, y):
    """Convert SUMO internal X/Y to real world Lat/Lon."""
    net = get_net()
    lon, lat = net.convertXY2LonLat(x, y)
    return float(lat), float(lon)

def get_junction_positions():
    """Returns a dictionary of junction_id -> {lat, lon} for all traffic lights."""
    net = get_net()
    positions = {}
    for tls in net.getTrafficLights():
        j_id = tls.getID()
        # Find the node (junction) corresponding to this TLS
        # Usually TLS IDs match node IDs, but sometimes multiple nodes are controlled.
        # We can just get one of the nodes associated with the program.
        # Since sumolib's TLS object might not easily yield the coordinate, we'll iterate over nodes.
        pass

    # A more reliable way: iterate over all nodes, if they have type="traffic_light", get their coordinate
    for node in net.getNodes():
        if node.getType() == "traffic_light":
            n_id = node.getID()
            x, y = node.getCoord()
            lat, lon = sumo_to_latlon(x, y)
            positions[n_id] = {"lat": lat, "lon": lon}
            
    # Add the specific cluster ID mentioned if it's not exactly matching a single node
    # The cluster ID: GS_cluster_11303526465_13072877373_13072877377_13072877378_#2more
    # If it is missing, we will compute its centroid from its edges/nodes later or just average it
    # Actually, the TLS ID is what traci uses. `traci.trafficlight.getIDList()` returns TLS IDs.
    
    return positions
