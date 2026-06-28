# SUMO Network Files

This directory holds SUMO network files generated via OSMWebWizard.

## How to Generate

```bash
# Navigate to your SUMO installation's tools/ directory
cd $SUMO_HOME/tools

# Launch OSMWebWizard (opens a browser)
python osmWebWizard.py
```

1. Draw your bounding box in the browser over your target area
2. Set vehicle demand (through-traffic + some local trips)
3. Click **Generate Scenario**
4. Copy the generated files here:
   - `osm.net.xml` — road network
   - `osm.rou.xml` — vehicle routes
   - `osm.sumocfg` — simulation config

## Emergency Vehicle Type

Add this to your `osm.rou.xml` (inside `<routes>`, before `<vehicle>` definitions):

```xml
<vType id="emergency"
       vClass="emergency"
       color="1,0,0"
       guiShape="emergency"
       maxSpeed="25"/>
```

## Quick Test

```bash
sumo-gui -c osm.sumocfg
```
