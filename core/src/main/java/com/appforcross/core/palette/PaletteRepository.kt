// Hash 173a2172ef28ea3268f28f95aa1c32f4
package com.appforcross.core.palette

interface PaletteRepository {
    fun list(): List<PaletteMeta>
    fun get(id: String): Palette?
}
