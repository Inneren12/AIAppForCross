package com.appforcross.editor.auto.photo.hq

/**
 * Пороговые правила приёмки + корректировка параметров на один авто‑ретрай.
 */
object AcceptanceGuard {
    data class Metrics(
        val banding: Float,
        val fsPct: Float,
        val singletonPct: Float,
        val runLen50: Float,
        val used: Int,
        val deltaE95: Float
    )

    fun collect(logs: Map<String, Any>): Metrics {
        fun f(key: String, def: Float) = (logs[key] as? Number)?.toFloat() ?: def
        fun i(key: String, def: Int) = (logs[key] as? Number)?.toInt() ?: def
        return Metrics(
            banding = f("banding", 0f),
            fsPct = f("fsPct", 0f),
            singletonPct = f("singletonPct", 0f),
            runLen50 = f("runLen50", 0f),
            used = i("usedSwatches", 0),
            deltaE95 = f("deltaE95", 0f)
        )
    }

    fun needRetry(preset: PresetGate.Preset, a: Metrics): Boolean = when (preset) {
        PresetGate.Preset.PORTRAIT_LIGHT,
        PresetGate.Preset.PORTRAIT_DARK,
        PresetGate.Preset.GENERAL ->
            a.singletonPct > 1.5f || a.runLen50 < 5f || a.fsPct > 0.25f || a.deltaE95 > 14f || a.used !in 22..30
        PresetGate.Preset.SKY_SEA,
        PresetGate.Preset.SNOW_NEUTRAL ->
            a.banding > 0.05f || a.fsPct > 0.25f || a.used < 30 || a.deltaE95 > 14f
        PresetGate.Preset.LOW_KEY ->
            a.fsPct > 0.25f || a.used < 26
    }

    fun adjust(p: PresetGate.Params): PresetGate.Params = p.copy(
        kMin = (p.kMin + 2).coerceAtMost(36),
        minAfterMerge = (p.minAfterMerge + 4).coerceAtMost(34),
        skinMin = (p.skinMin + 2).coerceAtMost(18),
        ampBase = (p.ampBase - 0.02f).coerceAtLeast(0.12f),
        ampStrong = (p.ampStrong + 0.02f).coerceAtMost(0.26f),
        fsTarget = (p.fsTarget - 0.03f).coerceAtLeast(0.18f)
    )
}