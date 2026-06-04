"""
Generate a 3x3 grid network for SUMO.
Command line usage: python generate_network.py

File structure(After running the script):
sumo_rl/
    nets/
        3x3grid/
            3x3.net.xml
            3x3.rou.xml
            3x3.sumocfg

Design choices:
    - 3x3 grid, but only the center intersection is controlled by the RL agent.(Have traffic light)
    - The other 8 intersections use right_before_left priority rule, don't need traffic light.
    - Each direction have 2 lanes: left turn only + straight and right turn only.
    - Have sidewalks and crossing for pedestrians, also controlled by traffic light.
    - Vehciles are generated using poisson distribution with probability of 0.1 for each lane, and pedestrians are generated with probability of 0.05 for each crossing.
"""
import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

# Path settings
SCRIPT_DIR = Path(__file__).parent
NET_FILE = SCRIPT_DIR / "3x3.net.xml"
ROU_FILE = SCRIPT_DIR / "3x3.rou.xml"
SUMOCFG_FILE = SCRIPT_DIR / "3x3.sumocfg"

# Parameters settings
GRID_SIZE = 3 # 3x3 grid
EDGE_LENGTH = 200 # length of each edge in meters
LANES_PER_DIRECTION = 2 # left turn only + straight and right turn only

SPEED_LIMIT = 13.89 # 50 km/h in m/s
SIDEWALK_WIDTH = 2 # width of sidewalks in meters
SIM_DURATION = 3600 # simulation duration in seconds
VEHICLE_PROB = 0.1 # probability of vehicle generation for each lane
PEDESTRIAN_PROB = 0.05 # probability of pedestrian generation for each crossing
CENTER_TLS_ID = "B1" # traffic light id for the center intersection

def check_sumo_home():
    """
    Check if SUMO_HOME environment variable is set and valid.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME environment variable is not set. Please set it to the SUMO installation directory.")
    netgenerate = Path(sumo_home) / "bin" / "netgenerate"
    if not netgenerate.is_file():
        raise FileNotFoundError(f"SUMO netgenerate tool not found at {netgenerate}. Please check your SUMO installation.")
    return sumo_home

def generate_net():
    """
    Use netgenerate to create a 3x3 grid network with the specified parameters.

    Key parameters for netgenerate:
    * --grid.x-number 3, --grid.y-number 3
        generate a 3x3 grid network, total 9 intersections.
        -> But only the center intersection need RL control.
    
    * --lanes 2
        each direction have 2 lanes: 
        lane 0: left turn only
        lane 1: straight and right turn only
    
    * --sidewalk-width 2.0
        add sidewalks with width of 2 meters on both sides of the roads.
    
    * --crossing true
        Generate crossing in intersection for pedestrians, also controlled by traffic light.

    * --walkingareas true
        Generate waiting areas in four corners of the intersection for pedestrians to wait before crossing.
    
    * --tls.set "A1A1"
        Only specify the center intersection (A1A1) to have traffic light, the other 8 intersections will use right_before_left priority rule.
        Because we only wnat RL to control the center intersection.
    
    * --default.junctions.keep-clear false
        Allow the vehicles to stop in the junction area, which is necessary for the right_before_left priority rule to work properly.
    """
    netgenerate_cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number", str(GRID_SIZE),
        "--grid.y-number", str(GRID_SIZE),
        "--grid.x-length", str(EDGE_LENGTH),
        "--grid.y-length", str(EDGE_LENGTH),
        "--default.lanenumber", str(LANES_PER_DIRECTION),
        "--default.speed", str(SPEED_LIMIT),
        "--default.sidewalk-width", str(SIDEWALK_WIDTH),
        "--sidewalks.guess","true",
        "--crossings.guess","true",
        "--walkingareas","true",
        "--tls.guess","false",
        "--tls.set", CENTER_TLS_ID,
        "--default.junctions.keep-clear", "0",
        "--output-file", str(NET_FILE),
        "--no-warnings","true"
    ]
    
    print("Generating network with netgenerate...")
    result = subprocess.run(netgenerate_cmd, capture_output=True, text=True, check=True)

    if result.returncode != 0:
        print(f"Error generating network: {result.stderr}")
        raise RuntimeError("Failed to generate network with netgenerate.")
    else:
        print("Network generated successfully.")
    


def generate_routes():
    """
    Generate vehicles and pedestrians route file (.rou.xml).

    Vehicles settings:
        - Use <flow> element to generate vehicles with poisson distribution.
        - Include: N->S, S->N, E->W, W->E, and also left turn flows: N->E, S->W, E->S, W->N.
        - Use vehsPerHour attribute to specify the average number of vehicles generated per hour for each flow, which can be calculated from the probability of vehicle generation for each lane.
    
    Pedestrians settings:
        - Use <personFlow> element to generate pedestrians with poisson distribution.
        - Include crossing flows for all four directions at the center intersection (A1A1): N->S, S->N, E->W, W->E.
        - Use personsPerHour attribute to specify the average number of pedestrians generated per hour for each flow, which can be calculated from the probability of pedestrian generation for each crossing.
        - Use <walk> to specify the edge.
    
    Edge ID rules(May be different in your generated network, please check the generated .net.xml file to confirm the edge IDs):
        - For the center intersection (A1A1), the incoming edges are: 
            N->C: A1A2/A1A1, S->C: A1A0/A1A1, E->C: A2A1/A1A1, W->C: A0A1/A1A1
        - Outgoing edges are:
            C->N: A1A1/A1A2, C->S: A1A1/A1A0, C->E: A1A1/A2A1, C->W: A1A1/A0A1
    """
    print("Generating routes file...")
    
    center = CENTER_TLS_ID
    c = GRID_SIZE // 2 # center index = 1

    # Edge naming: netgenerate grid format is "A{col}{row}/A{col}{row}"
    # Buttom left corner is (0,0), top right corner is (2,2)
    north_in = f"B{c+1}{center}" # From north to center
    south_in = f"B{c-1}{center}" # From south to center
    east_in = f"C{c}{center}" # From east to center
    west_in = f"A{c}{center}" # From west to center

    north_out = f"{center}B{c+1}" # From center to north
    south_out = f"{center}B{c-1}" # From center to south
    east_out = f"{center}C{c}" # From center to east
    west_out = f"{center}A{c}" # From center to west

    # Use python to generate the xml file for routes
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    # Car tyoe definition
    vtype = ET.SubElement(root, "vType") # Generate <> element for vehicle type definition
    vtype.set("id", "car")
    vtype.set("accel", "2.6") # acceleration in m/s^2
    vtype.set("decel", "4.5") # deceleration in m/s^2
    vtype.set("sigma", "0.5") # driver imperfection
    vtype.set("length", "5.0") # vehicle length in meters
    vtype.set("maxSpeed", str(SPEED_LIMIT)) # max speed in m/s
    vtype.set("minGap", "2.5") # minimum gap to the front vehicle in meters
    vtype.set("guiShape", "passenger") # shape for visualization

    # Pedestrian type definition
    ptype = ET.SubElement(root, "vType") # Generate <> element for pedestrian
    ptype.set("id", "pedestrian")
    ptype.set("vClass", "pedestrian")
    ptype.set("minGap", "0.25") # minimum gap to the front pedestrian in meters
    ptype.set("maxSpeed", "1.39") # max speed in m/s(About 5 km/h)

    # Vehicle flows

    vehicle_routes = [
        ("flow_NS", north_in, south_out, "N->S straight"), # N->S
        ("flow_SN", south_in, north_out, "S->N straight"), # S->N
        ("flow_EW", east_in, west_out, "E->W straight"), # E->W
        ("flow_WE", west_in, east_out, "W->E straight"), # W->E
        ("flow_NE", north_in, east_out, "N->E left turn"), # N->E left turn
        ("flow_SW", south_in, west_out, "S->W left turn"), # S->W left turn
        ("flow_ES", east_in, south_out, "E->S left turn"), # E->S left turn
        ("flow_WN", west_in, north_out, "W->N left turn"), # W->N left turn
        ("flow_NW", north_in, west_out, "N->W right turn"), # N->W right turn
        ("flow_SE", south_in, east_out, "S->E right turn"), # S-> E right turn
    ]

    for flow_id, from_edge, to_edge, description in vehicle_routes:
        flow = ET.SubElement(root, "flow")
        flow.set("id", flow_id)
        flow.set("type", "car")
        flow.set("from", from_edge)
        flow.set("to", to_edge)
        flow.set("begin", "0")
        flow.set("end", str(SIM_DURATION))
        flow.set("vehsPerHour", str(VEHICLE_PROB * 3600)) # Convert probability to vehicles per hour
        flow.set("departLane", "best") # Let SUMO choose the best lane for departure
        flow.set("departSpeed", "max") # Depart with max speed

    # Pedestrian flows
    ped_crossing = [
        # (flow_id, from_edge, to_edge, description)
        ("ped_NS", north_in, south_out, "Pedestrian crossing N->S"), # N->S crossing
        ("ped_SN", south_in, north_out, "Pedestrian crossing S->N"), # S->N crossing
        ("ped_EW", east_in, west_out, "Pedestrian crossing E->W"), # E->W crossing
        ("ped_WE", west_in, east_out, "Pedestrian crossing W->E") # W->E crossing
    ]

    for flow_id, from_edge, to_edge, description in ped_crossing:
        person_flow = ET.SubElement(root, "personFlow")
        person_flow.set("id", flow_id)
        person_flow.set("type", "pedestrian")
        person_flow.set("begin", "0")
        person_flow.set("end", str(SIM_DURATION))
        person_flow.set("personsPerHour", str(PEDESTRIAN_PROB * 3600)) # Convert probability to persons per hour

        walk = ET.SubElement(person_flow, "walk")
        walk.set("from", from_edge)
        walk.set("to", to_edge)
    
    # Output the xml to .rou.xml file
    _write_preety_xml(root, ROU_FILE)
    print("Routes file generated successfully.")

def _write_preety_xml(root: ET.Element, file_path: Path):
    """
    Write the XML tree to a file with pretty formatting.
    """
    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8")
    with open(file_path, "wb") as f:
        f.write(pretty_xml)

def generate_sumocfg():
    """
    Generate the sumocfg file (.sumocfg).

    .sumocfg is SUMO main configuration file, which specifies the input files and simulation settings.
    Use command traci.start([sumo, "-c", "3x3.sumocfg"]) to start the simulation with the generated sumocfg file.
    """
    print("Generating sumocfg file...")
    sumocfg = ET.Element("configuration")

    # Input files settings
    input_elem = ET.SubElement(sumocfg, "input")
    
    net_elem = ET.SubElement(input_elem, "net-file")
    net_elem.set("value", "3x3.net.xml")
    
    rou_elem = ET.SubElement(input_elem, "route-files")
    rou_elem.set("value", "3x3.rou.xml")

    # time settings
    time_elem = ET.SubElement(sumocfg, "time")
    begin = ET.SubElement(time_elem, "begin")
    begin.set("value", "0")
    end = ET.SubElement(time_elem, "end")
    end.set("value", str(SIM_DURATION))
    step_length = ET.SubElement(time_elem, "step-length")
    step_length.set("value", "1") # simulation step length in seconds

    # processing settings
    processing_elem = ET.SubElement(sumocfg, "processing")
    wt = ET.SubElement(processing_elem, "waiting-time-memory")
    wt.set("value", "10000") # maximum waiting time for vehicles in seconds

    # Report settings
    report_elem = ET.SubElement(sumocfg, "report")
    no_step_log = ET.SubElement(report_elem, "no-step-log")
    no_step_log.set("value", "true") # disable step log to reduce output
    no_warnings = ET.SubElement(report_elem, "no-warnings")
    no_warnings.set("value", "true") # disable warnings to reduce output

    _write_preety_xml(sumocfg, SUMOCFG_FILE)
    print("Sumocfg file generated successfully.")

def main():
    check_sumo_home()
    generate_net()
    generate_routes()
    generate_sumocfg()

    print("=" * 55)
    print("All files generated successfully.")
    print("=" * 55)

if __name__ == "__main__":
    main()