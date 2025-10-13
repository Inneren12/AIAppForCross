//Hash 7459a347d10dd7437a7687eef27f134f
package com.appforcross.app.palette

import android.content.Context
import com.appforcross.core.palette.Palette
import com.appforcross.core.palette.PaletteMeta
import com.appforcross.core.palette.PaletteRepository
import com.appforcross.core.palette.Swatch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

class AndroidPaletteRepository(private val appContext: Context) : PaletteRepository {

    private val json = Json { ignoreUnknownKeys = true }
    private val builtins = listOf(
        "palettes/dmc.json",
        "palettes/anchor.json",
        "palettes/toho.json",
        "palettes/preciosa.json"
    )

    // ----- DTO только для :app (соответствуют JSON) -----
    @Serializable
    private data class PalColorDto(
        val id: Int? = null,
        val code: String,
        val name: String,
        val rgb: String
    )

    @Serializable
    private data class PaletteDto(
        val id: String,
        val name: String,
        val type: String,
        val colors: List<PalColorDto>
    )

    // ----- Public API (core) -----

    override fun list(): List<PaletteMeta> = builtins.mapNotNull { path ->
        readText(path)?.let { raw ->
            runCatching { json.decodeFromString(PaletteDto.serializer(), raw) }.getOrNull()
        }
    }.map { dto ->
        PaletteMeta(
            id = dto.id,
            name = dto.name,
            colors = dto.colors.size
        )
    }

    override fun get(id: String): Palette? {
        val path = when (id) {
            "dmc" -> "palettes/dmc.json"
            "anchor" -> "palettes/anchor.json"
            "toho" -> "palettes/toho.json"
            "preciosa" -> "palettes/preciosa.json"
            else -> null
        } ?: return null

        val raw = readText(path) ?: return null
        val dto = runCatching { json.decodeFromString(PaletteDto.serializer(), raw) }.getOrNull() ?: return null

        return Palette(
            id = dto.id,
            name = dto.name,
            colors = dto.colors.map { c ->
                Swatch(
                    code = c.code,
                    name = c.name,
                    argb = parseHexRgb(c.rgb)
                )
            }
        )
    }

    // ----- Helpers -----

    private fun readText(path: String): String? = try {
        appContext.assets.open(path).use { String(it.readBytes(), Charsets.UTF_8) }
    } catch (_: Exception) { null }

    private fun parseHexRgb(rgb: String): Int {
        // Поддержка форматов "#RRGGBB" или "RRGGBB"
        val s = rgb.trim().removePrefix("#")
        require(s.length == 6) { "RGB must be 6 hex chars: $rgb" }
        val r = s.substring(0, 2).toInt(16)
        val g = s.substring(2, 4).toInt(16)
        val b = s.substring(4, 6).toInt(16)
        val a = 0xFF
        return (a shl 24) or (r shl 16) or (g shl 8) or b
    }
}
