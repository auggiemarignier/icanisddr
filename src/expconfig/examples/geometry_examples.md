# Example Geometry Configurations

## Example 1: Simple inner core and outer core

```yaml
regions:
  - type: ball
    radius: 1221.5
    label: IC
  - type: shell
    radius_inner: 1221.5
    radius_outer: 3480.0
    label: OC
```

## Example 2: Hemispheric inner core

```yaml
regions:
  - type: hemisphere
    radius: 1221.5
    normal: [0.0, 0.0, 1.0]
    label: IC_north
  - type: hemisphere
    radius: 1221.5
    normal: [0.0, 0.0, -1.0]
    label: IC_south
```

## Example 3: Layered inner core structure

```yaml
regions:
  - type: ball
    radius: 500.0
    label: IC_inner
  - type: shell
    radius_inner: 500.0
    radius_outer: 1000.0
    label: IC_middle
  - type: shell
    radius_inner: 1000.0
    radius_outer: 1221.5
    label: IC_outer
```

## Usage in Python

```python
import yaml
from expconfig import GeometryConfig

# Load from YAML
with open("geometry.yaml") as f:
    config_dict = yaml.safe_load(f)

geometry_config = GeometryConfig(**config_dict)

# Convert to raytracer objects
composite_geometry = geometry_config.to_composite_region()

# Use with rays
from raytracer import Ray
import numpy as np

entry = np.array([[1000.0, 0.0, 0.0]])
exit = np.array([[-800.0, 500.0, 300.0]])

ray = Ray(entry, exit)
distances = composite_geometry.ray_distances(ray.origin, ray.direction)

print(f"Total distance through all regions: {distances}")
print(f"Region labels: {composite_geometry.labels}")
```

## Convenience constructors

```python
from expconfig import GeometryConfig

# Standard Earth IC + OC
earth_config = GeometryConfig.earth_imic(
    imic_radius=650.0,
    ic_radius=1221.5,
)

# Hemispheric division
hemispheric_config = GeometryConfig.hemispheric_ic(
    ic_radius=1221.5,
    normal=[1.0, 0.0, 0.0],  # x-axis division
)
```
