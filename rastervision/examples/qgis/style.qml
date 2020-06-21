<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis minScale="1e+8" version="3.0.2-Girona" maxScale="0" hasScaleBasedVisibilityFlag="0">
  <pipe>
    <rasterrenderer type="paletted" alphaBand="-1" band="1" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry color="#ca3c3c" value="0" label="0" alpha="255"/>
        <paletteEntry color="#ff7c01" value="1" label="Building" alpha="127"/>
        <paletteEntry color="#2e31da" value="2" label="Background" alpha="0"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]"/>
    </rasterrenderer>
    <brightnesscontrast contrast="0" brightness="0"/>
    <huesaturation grayscaleMode="0" colorizeBlue="128" saturation="0" colorizeGreen="128" colorizeRed="255" colorizeStrength="100" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
