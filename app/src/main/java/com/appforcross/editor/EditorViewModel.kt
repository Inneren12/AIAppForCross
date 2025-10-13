// Hash 3b7a406bdac60532aedb1285442c492c
package com.appforcross.editor

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import android.content.Context
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.asImageBitmap
import com.appforcross.editor.engine.EditorEngine
import com.appforcross.editor.model.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import androidx.compose.ui.graphics.ImageBitmap
import com.appforcross.core.palette.PaletteMeta
import com.appforcross.editor.auto.detect.SmartSceneDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.min
import kotlinx.coroutines.Job
import com.appforcross.editor.auto.AutoProcessor
import com.appforcross.editor.util.Perf
import android.util.Log

class EditorViewModel(private val engine: EditorEngine) : ViewModel() {

    private val _state = MutableStateFlow(EditorState())
    val state = _state.asStateFlow()

    // --- История (Undo/Redo) ---
private data class PipelineSnapshot(
        val appliedImport: ImageBitmap?,
        val appliedPreprocess: ImageBitmap?,
        val appliedSize: ImageBitmap?,
        val appliedPalette: ImageBitmap?,
        val appliedOptions: ImageBitmap?,
        val state: EditorState,
        val threads: List<ThreadItem>,
        val symbolDraft: Map<String, Char>,
        val symbolsPreview: Map<String, Char>,
        val activePaletteId: String
    )
    private val history = ArrayList<PipelineSnapshot>(32)
    private var cursor = -1
    private val MAX_HISTORY = 30
    private val _canUndo = MutableStateFlow(false)
    private val _canRedo = MutableStateFlow(false)
    val canUndo = _canUndo.asStateFlow()
    val canRedo = _canRedo.asStateFlow()
    private var autoJob: Job? = null
    // Тонкий интерактор без UI-зависимостей — единая точка авто‑обработки
    private val autoProcessor = AutoProcessor()
    private fun updateHistoryFlags() {
        _canUndo.value = cursor > 0
        _canRedo.value = cursor >= 0 && cursor < history.size - 1
    }
    private fun captureSnapshot(next: EditorState) = PipelineSnapshot(
        appliedImport, appliedPreprocess, appliedSize, appliedPalette, appliedOptions,
        next,
        _threads.value,
        _symbolDraft.value,
        _symbolsPreview.value,
        getActivePaletteId()
    )
    private fun pushSnapshot(next: EditorState) {
        // Если делали Undo — срезаем хвост
        if (cursor < history.lastIndex) {
            history.subList(cursor + 1, history.size).clear()
        }
        history.add(captureSnapshot(next))
        // Ограничиваем длину истории
        if (history.size > MAX_HISTORY) {
            val drop = history.size - MAX_HISTORY
            repeat(drop) { history.removeAt(0) }
        }
        cursor = history.lastIndex
        updateHistoryFlags()
    }
    private fun applySnapshot(s: PipelineSnapshot) {
        appliedImport = s.appliedImport
        appliedPreprocess = s.appliedPreprocess
        appliedSize = s.appliedSize
        appliedPalette = s.appliedPalette
        appliedOptions = s.appliedOptions
        _threads.value = s.threads
        _symbolDraft.value = s.symbolDraft
        _symbolsPreview.value = s.symbolsPreview
        _activePaletteId.value = s.activePaletteId
        _state.value = s.state.copy(isBusy = false, error = null)
    }
    fun undo() {
        if (cursor <= 0) return
        cursor--
        applySnapshot(history[cursor])
        updateHistoryFlags()
    }
    fun redo() {
        if (cursor < 0 || cursor >= history.size - 1) return
        cursor++
        applySnapshot(history[cursor])
        updateHistoryFlags()
    }

    // --- Пайплайн: зафиксированные изображения по стадиям ---
    private var appliedImport: ImageBitmap? = null
    private var appliedPreprocess: ImageBitmap? = null
    private var appliedSize: ImageBitmap? = null
    private var appliedPalette: ImageBitmap? = null
    private var appliedOptions: ImageBitmap? = null

    // dirty-флаги стадий (ниже по графу)
    private var dirtySize = false
    private var dirtyPalette = false
    private var dirtyOptions = false

    // Хранилище палитр: нужно для computeThreadStatsAgainstPalette
    private var paletteRepository: com.appforcross.core.palette.PaletteRepository? = null
    fun setPaletteRepository(repo: com.appforcross.core.palette.PaletteRepository) {
        paletteRepository = repo
    }

    // ── Авто‑распознавание при импорте и подсказка типа сцены (до обработки)
    private val _autoDetectOnImport = MutableStateFlow(true)
    val autoDetectOnImport = _autoDetectOnImport.asStateFlow()
    fun setAutoDetectOnImport(v: Boolean) { _autoDetectOnImport.value = v }

    // Авто-обработка после импорта (V2/V1)
    private val _autoProcessOnImport = MutableStateFlow(true)
    val autoProcessOnImport = _autoProcessOnImport.asStateFlow()
    fun setAutoProcessOnImport(v: Boolean) { _autoProcessOnImport.value = v }

    data class ImportSceneHint(
        val kind: String,
        val confidence: Float,
        val top3: List<Pair<String, Float>>,
        val widthStitches: Int? = null,
        val colorsSelected: Int? = null
    )
    private val _importSceneHint = MutableStateFlow<ImportSceneHint?>(null)
    val importSceneHint = _importSceneHint.asStateFlow()
    // ── Короткая сводка для Import: ширина в стежках (S) и число цветов (K)
    data class AutoSummary(val widthSt: Int, val colors: Int)
    private val _autoSummary = MutableStateFlow<AutoSummary?>(null)
    val autoSummary: kotlinx.coroutines.flow.StateFlow<AutoSummary?> = _autoSummary.asStateFlow()

    // Активная палитра — реактивно (StateFlow), без изменения EditorState
    private val _activePaletteId = MutableStateFlow("")
    val activePaletteId: kotlinx.coroutines.flow.StateFlow<String> = _activePaletteId.asStateFlow()
    fun getPalettes(): List<PaletteMeta> = paletteRepository?.list().orEmpty()
    fun getActivePaletteId(): String {
        val cached = _activePaletteId.value
        if (cached.isNotEmpty()) return cached
        val id = getPalettes().firstOrNull()?.id ?: "dmc"
        _activePaletteId.value = id
        return id
    }
    fun setActivePalette(id: String) { _activePaletteId.value = id }

    // Модель нитки для UI (локально, без правок EditorState)
    data class ThreadItem(
        val code: String,
        val name: String,
        val argb: Int,
        val percent: Int,
        val count: Int
    )
    private val _threads = MutableStateFlow<List<ThreadItem>>(emptyList())
    val threads: kotlinx.coroutines.flow.StateFlow<List<ThreadItem>> = _threads.asStateFlow()


       // --- Символы ниток: draft (редактирование) и применённые к предпросмотру ---
       private val symbolSet: CharArray = charArrayOf(
           '●','○','■','□','▲','△','◆','◇','★','☆','✚','✖','✳','◼','◻','✦','✧',
           'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
           'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
           '0','1','2','3','4','5','6','7','8','9'
       )
    private val _symbolDraft = MutableStateFlow<Map<String, Char>>(emptyMap())
    val symbolDraft = _symbolDraft.asStateFlow()
    private val _symbolsPreview = MutableStateFlow<Map<String, Char>>(emptyMap())
    val symbolsPreview = _symbolsPreview.asStateFlow()
    fun symbolFor(code: String): Char? = _symbolDraft.value[code] ?: _symbolsPreview.value[code]
    fun setSymbol(code: String, ch: Char) {
        val cur = _symbolDraft.value.toMutableMap()
        // уникальность: если символ занят другим цветом — перекинем тот цвет на свободный
        val conflict = cur.entries.firstOrNull { it.value == ch && it.key != code }?.key
        if (conflict != null) {
            val used = cur.values.toMutableSet()
            used.add(ch)
            cur[conflict] = pickFreeSymbol(used)
        }
        cur[code] = ch
        _symbolDraft.value = cur
    }
    private fun pickFreeSymbol(used: MutableSet<Char>): Char {
        for (c in symbolSet) if (used.add(c)) return c
        return '?'
    }
    private fun autoAssignSymbolsForThreads(list: List<ThreadItem>) {
        val cur = _symbolDraft.value.toMutableMap()
        val used = cur.values.toMutableSet()
        list.forEach { item ->
            if (!cur.containsKey(item.code)) {
                cur[item.code] = pickFreeSymbol(used)
                            }
        }
        _symbolDraft.value = cur
    }
    fun hasSymbolsFor(codes: List<String>): Boolean {
        val m = if (_symbolDraft.value.isNotEmpty()) _symbolDraft.value else _symbolsPreview.value
        return codes.all { m.containsKey(it) }
    }
    fun commitSymbolsForPreview() { _symbolsPreview.value = _symbolDraft.value }
    fun getActivePaletteSwatches(): List<com.appforcross.core.palette.Swatch> =
        paletteRepository?.get(getActivePaletteId())?.colors.orEmpty()

    // --- Авто-символы: реактивный флаг и «умный» автоподбор ---
    private val _autoSymbolsEnabled = MutableStateFlow(true)
    val autoSymbolsEnabled = _autoSymbolsEnabled.asStateFlow()
    fun setAutoSymbolsEnabled(enabled: Boolean) { _autoSymbolsEnabled.value = enabled }

    fun autoAssignSymbolsIfEnabled() {
        // Если автоподбор выключен — просто зафиксировать текущий черновик
        if (!_autoSymbolsEnabled.value) {
            commitSymbolsForPreview()
            return
        }
        val list = _threads.value
        if (list.isEmpty()) {
            commitSymbolsForPreview()
            return
        }
        val assigned = autoAssignSymbolsSmart(list, symbolSet.asList())
        val cur = _symbolDraft.value.toMutableMap()
        assigned.forEach { (code, ch) -> cur[code] = ch }
        _symbolDraft.value = cur
        commitSymbolsForPreview()
    }

    // Предпочтения по «плотности» глифов и набор проверенных символов
    private val densePref = charArrayOf('●','■','◆','▲','★','◼','✦','✚','✖','✳')
    private val openPref  = charArrayOf('○','□','◇','△','☆','◻','✧')
    private val crossPref = charArrayOf('✚','✖','✳','✦','✧')
    private val ambiguous = setOf('O','0','I','l','1') // избегаем спорных символов

    private fun luma01(argb: Int): Double {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    }

    /** Умный автоподбор символов: контраст/заполненность/разнообразие */
    private fun autoAssignSymbolsSmart(
        threads: List<ThreadItem>,
        allowedSymbols: List<Char>
    ): Map<String, Char> {
        val allowed = allowedSymbols.filterNot { it in ambiguous }.toMutableList()
        // Списки-кандидаты по категориям (фильтрация по разрешённым)
        val dense = densePref.filter { it in allowed }
        val open  = openPref.filter  { it in allowed }
        val cross = crossPref.filter  { it in allowed }
        // Алфавит/цифры на добивку
        val alphaNum = allowed.filter { it.isLetterOrDigit() }

        val used = HashSet<Char>(threads.size)
        val result = LinkedHashMap<String, Char>(threads.size)
        // Больше стежков → важнее читаемость
        val ordered = threads.sortedByDescending { it.count }
        for (t in ordered) {
            val y = luma01(t.argb)
            // Светлые клетки → тёмные/«плотные» глифы; тёмные клетки → «полые».
            val pref = when {
                y >= 0.70 -> sequenceOf(dense, cross, open, alphaNum)
                y <= 0.35 -> sequenceOf(open, cross, dense, alphaNum)
                else      -> sequenceOf(cross, dense, open, alphaNum)
            }.flatten()
            val chosen = pref.firstOrNull { it !in used } ?: allowed.firstOrNull { it !in used } ?: '?'
            used.add(chosen)
            result[t.code] = chosen
        }
        return result
    }

    // --- Экспорт: пост-действия (Preview/Import)
    private val _lastExportUri = MutableStateFlow<Uri?>(null)
    val lastExportUri = _lastExportUri.asStateFlow()
    private val _exports = MutableStateFlow<List<Uri>>(emptyList())
    val exports = _exports.asStateFlow()
    fun registerExport(uri: Uri) {
        _lastExportUri.value = uri
        _exports.value = (listOf(uri) + _exports.value).distinct().take(10)
    }
    fun deleteExport(ctx: Context, uri: Uri): Boolean {
        val ok = DocumentFile.fromSingleUri(ctx, uri)?.delete() == true
        if (ok) {
            if (_lastExportUri.value == uri) _lastExportUri.value = null
            _exports.value = _exports.value.filterNot { it == uri }
        }
        return ok
    }

    // --- Edge‑prefilter toggle (без изменения EditorState) ---
    private val _edgeEnhance = MutableStateFlow(false)
    val edgeEnhanceEnabled = _edgeEnhance.asStateFlow()
    fun setEdgeEnhanceEnabled(enabled: Boolean) { _edgeEnhance.value = enabled }

    fun setSource(image: ImageBitmap, aspect: Float) {
        // Import.applied = исходник; сбрасываем нижние стадии
        appliedImport = image
        appliedPreprocess = null
        appliedSize = null
        appliedPalette = null
        appliedOptions = null
        dirtySize = false; dirtyPalette = false; dirtyOptions = false
        _state.value = _state.value.copy(sourceImage = image, previewImage = image, aspect = aspect)
        // начальная точка истории
        pushSnapshot(_state.value)
        // После импорта — либо полный авто‑прогон, либо только быстрый hint
        if (_autoProcessOnImport.value) {
            viewModelScope.launch {
                runCatching { runAutoOnImportIfNeeded(image.asAndroidBitmap()) }
                    .onFailure {
                        _state.value = _state.value.copy(isBusy = false, error = it.message)
                        android.util.Log.w("EditorVM", "auto process after import failed", it)
                    }
            }
        } else if (_autoDetectOnImport.value) {
            viewModelScope.launch {
                runCatching { detectSceneForImport() }
                    .onFailure {
                        _state.value = _state.value.copy(error = it.message)
                        android.util.Log.w("EditorVM", "detectSceneForImport() failed", it)
                    }
            }
        } else {
            _importSceneHint.value = null
            _autoSummary.value = null
        }
    }

    fun updatePreprocess(transform: (PreprocessState) -> PreprocessState) {
        _state.value = _state.value.copy(preprocess = transform(_state.value.preprocess))
    }

    fun applyPreprocess() = runStage { st ->
        val base = appliedImport ?: st.sourceImage ?: return@runStage st
        val out = engine.applyPreprocess(base, st.preprocess)
        // фиксируем стадию + инвалидируем последующие
        appliedPreprocess = out
        dirtySize = true; dirtyPalette = true; dirtyOptions = true
        val next = st.copy(previewImage = out)
        pushSnapshot(next)
        next
    }

    // Считаем «нитки» строго по фактически использованным swatch’ам результата.
    // EditorViewModel.kt
    private fun computeThreadsFromUsed(out: AutoProcessor.Output): List<ThreadItem> {
        val bmp = out.image.asImageBitmap()
        val ab = bmp.asAndroidBitmap()
        val w = ab.width
        val h = ab.height
        val total = (w * h).coerceAtLeast(1)
        val px = IntArray(total)
        ab.getPixels(px, 0, w, 0, 0, w, h)

        // База только из фактически использованных конвейером swatch’ей;
        // фоллбек — ТОЛЬКО если список пуст.
        val baseline =
            if (out.usedSwatches.isNotEmpty()) out.usedSwatches else getActivePaletteSwatches()
        android.util.Log.d(
            "EditorVM.Threads",
            "computeThreadsFromUsed: baseline=${baseline.size}, img=${w}x${h}"
        )
        if (baseline.isEmpty()) return emptyList()

        // Быстрый доступ: ARGB -> индекс swatch в baseline
        val argbToIdx = HashMap<Int, Int>(baseline.size * 2)
        baseline.forEachIndexed { i, sw -> argbToIdx[sw.argb or (0xFF shl 24)] = i }

        // Подсчитать, к какому из baseline ближе каждый пиксель (точное совпадение -> ближайший)
        val counts = IntArray(baseline.size)
        var idx = 0
        while (idx < total) {
            val c = px[idx] or (0xFF shl 24)
            val j = argbToIdx[c]
            if (j != null) {
                counts[j]++
            } else {
                // ближайший по RGB среди baseline
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                var best = 0
                var bestD = Int.MAX_VALUE
                var k = 0
                while (k < baseline.size) {
                    val a = baseline[k].argb
                    val dr = r - ((a shr 16) and 0xFF)
                    val dg = g - ((a shr 8) and 0xFF)
                    val db = b - (a and 0xFF)
                    val d = dr * dr + dg * dg + db * db
                    if (d < bestD) {
                        bestD = d; best = k
                    }
                    k++
                }
                counts[best]++
            }
            idx++
        }

        // Сформировать список по baseline, убрать нули, отсортировать
        val result = baseline.mapIndexed { bi, sw ->
            val cnt = counts[bi]
            ThreadItem(
                code = sw.code,
                name = sw.name,
                argb = sw.argb,
                percent = if (total > 0) (cnt * 100) / total else 0,
                count = cnt
            )
        }.filter { it.count > 0 }.sortedByDescending { it.count }
        android.util.Log.d(
            "EditorVM.Threads",
            "Result[${result.size}]: " + result.joinToString { "${it.code}:${it.percent}%(${it.count})" }
        )
        return result
    }

    fun updateSize(transform: (SizeState) -> SizeState) {
        _state.value = _state.value.copy(size = transform(_state.value.size))
    }

    fun applySize() = runStage { st ->
        // База для Size: Preprocess.applied -> Import.applied
        val base = appliedPreprocess ?: appliedImport ?: st.sourceImage ?: return@runStage st
        val s = st.size

        val srcBmp = base.asAndroidBitmap()
        // Аспект источника (если в стейте невалиден — считаем по картинке)
        val aspect = if (st.aspect > 0f) st.aspect
        else (srcBmp.width.toFloat() / srcBmp.height.coerceAtLeast(1))

        // Рассчитываем целевые размеры в "крестиках"
        var w = s.widthStitches.coerceAtLeast(1)
        var h = s.heightStitches.coerceAtLeast(1)
        when (s.pick) {
            SizePick.BY_WIDTH -> {
                if (s.keepAspect) h = ((w / aspect) + 0.5f).toInt().coerceAtLeast(1)
            }
            SizePick.BY_HEIGHT -> {
                if (s.keepAspect) w = ((h * aspect) + 0.5f).toInt().coerceAtLeast(1)
            }
            SizePick.BY_DPI -> {
                // На этом этапе физическую величину не пересчитываем;
                // при сохранении пропорций обновляем сопряжённую величину.
                if (s.keepAspect) h = ((w / aspect) + 0.5f).toInt().coerceAtLeast(1)
            }
        }

        // Масштабирование: 1 крестик = 1 пиксель результата
        val scaled = Bitmap.createScaledBitmap(srcBmp, w, h, true)
        val out = scaled.asImageBitmap()
        // фиксируем стадию + инвалидируем последующие
        appliedSize = out
        dirtyPalette = true; dirtyOptions = true
        st.copy(previewImage = out, size = s.copy(widthStitches = w, heightStitches = h))
    }

    fun updatePalette(transform: (PaletteState) -> PaletteState) {
        _state.value = _state.value.copy(palette = transform(_state.value.palette))
    }

    fun applyPaletteKMeans() = runStage { st ->
        val base = appliedSize ?: appliedPreprocess ?: appliedImport ?: st.sourceImage ?: return@runStage st
        val bmp = base.asAndroidBitmap()
        val palette = getActivePaletteSwatches()
        val out = autoProcessor.process(bmp, palette)
        // Статистику ниток строим строго по тому, что вернул конвейер:
        val threads = computeThreadsFromUsed(out)
        _threads.value = threads
        autoAssignSymbolsForThreads(threads)
        commitSymbolsForPreview()
        appliedPalette = out.image.asImageBitmap()
        dirtyOptions = true
        _autoSummary.value = AutoSummary(widthSt = out.widthStitches, colors = out.usedSwatches.size)
        _importSceneHint.value = ImportSceneHint(
            kind = out.kind.name,
            confidence = out.confidence,
            top3 = listOf(
                "DISCRETE" to (out.scores[SmartSceneDetector.Mode.DISCRETE] ?: 0f),
                "PHOTO"    to (out.scores[SmartSceneDetector.Mode.PHOTO]    ?: 0f)
            ),
            widthStitches = out.widthStitches,
            colorsSelected = out.usedSwatches.size
        )
        val next = st.copy(previewImage = appliedPalette)
        pushSnapshot(next)
        next
    }

    fun updateOptions(transform: (OptionsState) -> OptionsState) {
        _state.value = _state.value.copy(options = transform(_state.value.options))
    }

    fun applyOptions() = runStage { st ->
        // База для Options: Palette.applied -> Size.applied -> Preprocess.applied -> Import.applied
        val base = appliedPalette ?: appliedSize ?: appliedPreprocess ?: appliedImport ?: st.sourceImage ?: return@runStage st
        val out = engine.applyOptions(base, st.options, st.palette.metric)
        appliedOptions = out
        val next = st.copy(previewImage = out)
        pushSnapshot(next)
        next
    }

    private fun uniqueColors(bmp: ImageBitmap): Int {
        val ab = bmp.asAndroidBitmap()
        val n = ab.width * ab.height
        val buf = IntArray(n)
        ab.getPixels(buf, 0, ab.width, 0, 0, ab.width, ab.height)
        return buf.toHashSet().size
    }

    // Подсчёт «ниток» относительно активной палитры (fallback — топ-N по hex)
    private fun computeThreadStatsAgainstPalette(
        bmp: androidx.compose.ui.graphics.ImageBitmap,
        limit: Int,
        paletteId: String
    ): List<ThreadItem> {
        val ab = bmp.asAndroidBitmap()
        val w = ab.width
        val h = ab.height
        val total = (w * h).coerceAtLeast(1)
        val px = IntArray(total)
        ab.getPixels(px, 0, w, 0, 0, w, h)

        val pal = paletteRepository?.get(paletteId)
        if (pal == null || pal.colors.isEmpty()) {
            android.util.Log.w(
                "EditorVM.Threads",
                "computeThreadStatsAgainstPalette: palette empty → HEX fallback, limit=$limit, img=${w}x${h}"
            )
            // Fallback: топ-N по #RRGGBB
            val counts = HashMap<Int, Int>(1024)
            for (c in px) {
                val rgb = c and 0x00FFFFFF
                counts[rgb] = (counts[rgb] ?: 0) + 1
            }
            val res = counts.entries
                .sortedByDescending { it.value }
                .take(limit.coerceAtLeast(1))
                .map { (rgb, cnt) ->
                val argb = 0xFF000000.toInt() or rgb
                val hex = String.format("#%06X", rgb)
                ThreadItem(
                    code = hex,
                    name = hex,
                    argb = argb,
                    percent = (cnt * 100) / total,
                    count = cnt
                )
            }
            android.util.Log.d(
                "EditorVM.Threads",
                "computeThreadStatsAgainstPalette(HEX) -> ${res.size} items: " + res.joinToString { it.code }
            )
            return res
        }
        val colors = pal.colors
        val n = colors.size
        if (n == 0) return emptyList()
        android.util.Log.d(
            "EditorVM.Threads",
            "computeThreadStatsAgainstPalette: palColors=$n, limit=$limit, img=${w}x${h}"
        )
        // 1) Сжимаем картинку до уникальных RGB → счётчики
        val uniq = HashMap<Int, Int>(min(total, 2048))
        var idx = 0
        while (idx < total) {
            val rgb = px[idx] and 0x00FFFFFF
            uniq[rgb] = (uniq[rgb] ?: 0) + 1
            idx++
        }
        // 2) Палитра в OKLab
        val swL = FloatArray(n)
        val swA = FloatArray(n)
        val swB = FloatArray(n)
        var i = 0
        while (i < n) {
            val a = colors[i].argb
            val lab = rgbToOkLab((a shr 16) and 0xFF, (a shr 8) and 0xFF, a and 0xFF)
            swL[i] = lab[0]; swA[i] = lab[1]; swB[i] = lab[2]
            i++
        }
        // 3) Для каждого уникального RGB — ближайший swatch по OKLab
        val counts = IntArray(n)
        for ((rgb, cnt) in uniq) {
            val r = (rgb shr 16) and 0xFF
            val g = (rgb shr 8) and 0xFF
            val b = rgb and 0xFF
            val lab = rgbToOkLab(r, g, b)
            var best = 0
            var bestD = Float.POSITIVE_INFINITY
            var k = 0
            while (k < n) {
                val dl = lab[0] - swL[k]
                val da = lab[1] - swA[k]
                val db = lab[2] - swB[k]
                val d = dl*dl + da*da + db*db
                if (d < bestD) { bestD = d; best = k }
                k++
            }
            counts[best] += cnt
        }
        val res = counts
            .mapIndexed { i, cnt -> i to cnt }
            .filter { it.second > 0 }
            .sortedByDescending { it.second }
            .take(limit.coerceAtLeast(1))
            .map { (i, cnt) ->
            val sw = colors[i]
            ThreadItem(
                code = sw.code,
                name = sw.name,
                argb = sw.argb,
                percent = (cnt * 100) / total,
                count = cnt
            )
        }
        android.util.Log.d(
            "EditorVM.Threads",
            "computeThreadStatsAgainstPalette(res)[${res.size}]: " +
                    res.joinToString { "${it.code}:${it.percent}%(${it.count})" }
        )
        return res
    }

    // --- Edge prefilter: лёгкий unsharp‑mask перед k‑means ---
    private fun edgePrefilterUnsharp(src: ImageBitmap, amount: Float = 0.6f): ImageBitmap {
        val ab = src.asAndroidBitmap()
        val w = ab.width; val h = ab.height
        if (w <= 0 || h <= 0) return src
        val n = w * h
        val px = IntArray(n)
        ab.getPixels(px, 0, w, 0, 0, w, h)
        val blur = blur3x3(px, w, h)
        val out = IntArray(n)
        var i = 0
        val amt = amount.coerceIn(0f, 1.5f)
        while (i < n) {
            val s = px[i]; val b = blur[i]
            val a = (s ushr 24) and 0xFF
            val sr = (s ushr 16) and 0xFF; val sg = (s ushr 8) and 0xFF; val sb = s and 0xFF
            val br = (b ushr 16) and 0xFF; val bg = (b ushr 8) and 0xFF; val bb = b and 0xFF
            val rr = clamp255(sr + ((sr - br) * amt).toInt())
            val gg = clamp255(sg + ((sg - bg) * amt).toInt())
            val bb2 = clamp255(sb + ((sb - bb) * amt).toInt())
            out[i] = (a shl 24) or (rr shl 16) or (gg shl 8) or bb2
            i++
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp.asImageBitmap()
    }
    private fun blur3x3(src: IntArray, w: Int, h: Int): IntArray {
        fun clamp(x: Int, lo: Int, hi: Int) = if (x < lo) lo else if (x > hi) hi else x
        val out = IntArray(src.size)
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                var sa = 0; var sr = 0; var sg = 0; var sb = 0; var cnt = 0
                var dy = -1
                while (dy <= 1) {
                    val yy = clamp(y + dy, 0, h - 1)
                    var dx = -1
                    while (dx <= 1) {
                        val xx = clamp(x + dx, 0, w - 1)
                        val c = src[yy * w + xx]
                        sa += (c ushr 24) and 0xFF
                        sr += (c ushr 16) and 0xFF
                        sg += (c ushr 8) and 0xFF
                        sb += c and 0xFF
                        cnt++
                        dx++
                    }
                    dy++
                }
                val idx = y * w + x
                out[idx] = ((sa / cnt) shl 24) or ((sr / cnt) shl 16) or ((sg / cnt) shl 8) or (sb / cnt)
                x++
            }
            y++
        }
        return out
    }
    private fun clamp255(v: Int) = if (v < 0) 0 else if (v > 255) 255 else v

    // --- OKLab conversion (локально, без :core зависимостей) ---
    private fun rgbToOkLab(r8: Int, g8: Int, b8: Int): FloatArray {
        fun srgbToLinear(c: Int): Double {
            val s = c / 255.0
            return if (s <= 0.04045) s / 12.92 else Math.pow((s + 0.055) / 1.055, 2.4)
        }
        val r = srgbToLinear(r8)
        val g = srgbToLinear(g8)
        val b = srgbToLinear(b8)
        val l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        val m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        val s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        val l_ = kotlin.math.cbrt(l)
        val m_ = kotlin.math.cbrt(m)
        val s_ = kotlin.math.cbrt(s)
        val L = (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_).toFloat()
        val A = (1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_).toFloat()
        val B = (0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_).toFloat()
        return floatArrayOf(L, A, B)
    }

    private fun runStage(block: (EditorState) -> EditorState) {
        viewModelScope.launch {
            _state.value = _state.value.copy(isBusy = true, error = null)
            try {
                val next = withContext(Dispatchers.Default) { block(_state.value) }
                _state.value = next.copy(isBusy = false)
            } catch (t: Throwable) {
                _state.value = _state.value.copy(isBusy = false, error = t.message)
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
// Авто-прогон после импорта: SmartScene → AutoProcessor → Threads
// ──────────────────────────────────────────────────────────────
    private fun runAutoOnImportIfNeeded(imported: Bitmap) {
        if (!_autoProcessOnImport.value) return
        autoJob?.cancel()
        autoJob = viewModelScope.launch {
            _state.value = _state.value.copy(isBusy = true, error = null)
            try {
                // 1) Ранний SmartScene-hint
                val decision = withContext(Dispatchers.Default) { SmartSceneDetector.detect(imported) }
                _importSceneHint.value = ImportSceneHint(
                    kind = decision.mode.name,
                    confidence = decision.confidence,
                    top3 = listOf(
                        "DISCRETE" to (decision.scores[SmartSceneDetector.Mode.DISCRETE] ?: 0f),
                        "PHOTO"    to (decision.scores[SmartSceneDetector.Mode.PHOTO]    ?: 0f)
                    )
                )
                // 2) Обработка через AutoProcessor
                val palette = getActivePaletteSwatches()
                val out = withContext(Dispatchers.Default) {
                    autoProcessor.process(imported, palette, decision)
                }

                // 3) Подсчёт ниток/символов — ПРИОРИТЕТНО по реально использованным swatch’ам
                val threads = withContext(Dispatchers.Default) {
                    val uniq = uniqueColors(out.image.asImageBitmap())
                    android.util.Log.i(
                        "EditorVM.Auto",
                        "Auto: usedSwatches=${out.usedSwatches.size}, uniqColors=$uniq, activePalette=${palette.size}"
                    )
                    if (out.usedSwatches.isNotEmpty()) {
                        computeThreadsFromUsed(out)
                    } else {
                        val limit = palette.size.coerceAtLeast(1)
                        android.util.Log.w(
                            "EditorVM.Threads",
                            "Fallback to palette mapping: limit=$limit, paletteId=${getActivePaletteId()}"
                        )
                        computeThreadStatsAgainstPalette(out.image.asImageBitmap(), limit, getActivePaletteId())
                        }
                }
                _threads.value = threads
                autoAssignSymbolsForThreads(threads)
                commitSymbolsForPreview()

                // 4) Публикация результата (тайминг)
                Perf.trace("EditorVM.Auto", "publish ${out.kind.name}") {
                    appliedPalette = out.image.asImageBitmap()
                    _autoSummary.value = AutoSummary(widthSt = out.widthStitches, colors = out.usedSwatches.size)
                    _importSceneHint.value = _importSceneHint.value?.copy(
                        kind = out.kind.name,
                        widthStitches = out.widthStitches,
                        colorsSelected = out.usedSwatches.size
                    )
                    val next = _state.value.copy(previewImage = appliedPalette)
                    pushSnapshot(next)
                    _state.value = next.copy(isBusy = false)
                }
            } catch (t: Throwable) {
                _state.value = _state.value.copy(isBusy = false, error = t.message)
                android.util.Log.w("EditorVM.Auto", "auto process failed", t)
            }
        }
    }


    // ──────────────────────────────────────────────────────────────
    // Детект для «Импорт»: SmartSceneDetector из :core (тип + top‑3)
    // ──────────────────────────────────────────────────────────────
    private suspend fun detectSceneForImport() {
        val img = state.value.sourceImage ?: return
        val bm = img.asAndroidBitmap()
        // 1) Детект в фоне + ранний hint (с таймингом)
        val decision = Perf.traceSuspend("EditorVM.Auto", "SmartScene.detect") {
            withContext(Dispatchers.Default) { SmartSceneDetector.detect(bm) }
        }
        val sc = decision.scores
        val hint = ImportSceneHint(
            kind = decision.mode.name,                  // DISCRETE | PHOTO
            confidence = decision.confidence,
            top3 = listOf(
                "DISCRETE" to (sc[SmartSceneDetector.Mode.DISCRETE] ?: 0f),
                "PHOTO"    to (sc[SmartSceneDetector.Mode.PHOTO]    ?: 0f)
            )
        )
        _importSceneHint.value = hint
    }
}