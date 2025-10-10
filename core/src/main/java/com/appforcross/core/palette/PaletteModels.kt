// Hash a0098eb1383356d31f9dbd0d8f68dc35
package com.appforcross.core.palette

data class PaletteMeta(val id: String, val name: String, val colors: Int) // пример
data class Palette(val id: String, val name: String, val colors: List<Swatch>)
data class Swatch(val code: String, val name: String, val argb: Int)