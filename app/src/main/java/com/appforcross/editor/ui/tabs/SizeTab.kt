//Hash 33324763b75c5ca1f0106a4d94bac580
package com.appforcross.editor.ui.tabs

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.DropdownMenu
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.model.*
import com.appforcross.editor.ui.components.ApplyBar
import com.appforcross.editor.ui.components.Section
import com.appforcross.i18n.LocalStrings
import com.appforcross.i18n.Strings
import androidx.compose.ui.Alignment

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SizeTab(vm: EditorViewModel) {
    val st by vm.state.collectAsState()
    val s = st.size
    val S = LocalStrings.current

    Column(Modifier.fillMaxSize()) {
        Section(S.size.title) {
            if (st.isBusy) {
                // Индикатор процесса пересчёта размеров (indeterminate)
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Spacer(Modifier.height(8.dp))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = s.unit == UnitMode.ST_PER_INCH,
                    onClick = { vm.updateSize { it.copy(unit = UnitMode.ST_PER_INCH) } },
                    label = { Text(S.size.stPerInch) }
                )
                FilterChip(
                    selected = s.unit == UnitMode.ST_PER_CM,
                    onClick = { vm.updateSize { it.copy(unit = UnitMode.ST_PER_CM) } },
                    label = { Text(S.size.stPerCm) }
                )
            }

            Spacer(Modifier.height(12.dp))

            var expanded by remember { mutableStateOf(false) }
            ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
                OutlinedTextField(
                    readOnly = true,
                    value = densityLabel(S, s.presetDensity, s.unit),
                    onValueChange = {},
                    label = { Text(S.size.preset) },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                    modifier = Modifier.menuAnchor().fillMaxWidth()
                )
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    listOf(11f, 12f, 14f, 16f, 18f).forEach { d ->
                        DropdownMenuItem(text = { Text(densityLabel(S, d, s.unit)) }, onClick = {
                            vm.updateSize { it.copy(presetDensity = d) }
                            expanded = false
                        })
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SegButton(S.size.byWidth, selected = s.pick == SizePick.BY_WIDTH) {
                    vm.updateSize { it.copy(pick = SizePick.BY_WIDTH) }
                }
                SegButton(S.size.byHeight, selected = s.pick == SizePick.BY_HEIGHT) {
                    vm.updateSize { it.copy(pick = SizePick.BY_HEIGHT) }
                }
                SegButton(S.size.byDpi, selected = s.pick == SizePick.BY_DPI) {
                    vm.updateSize { it.copy(pick = SizePick.BY_DPI) }
                }
            }

            Spacer(Modifier.height(12.dp))

            when (s.pick) {
                SizePick.BY_WIDTH -> {
                    NumberField(
                        label = S.size.widthStitches,
                        value = s.widthStitches.toString(),
                        onValue = { v -> v.toIntOrNull()?.let { k -> vm.updateSize { s0 -> s0.copy(widthStitches = itBound(k)) } }
                        }
                    )
                    Spacer(Modifier.height(8.dp))
                    ReadOnlyLine(S.size.heightStitches, estimateHeight(s, st.aspect).toString())
                }
                SizePick.BY_HEIGHT -> {
                    NumberField(
                        label = S.size.heightStitches,
                        value = s.heightStitches.toString(),
                        onValue = { v -> v.toIntOrNull()?.let { k -> vm.updateSize { s0 -> s0.copy(heightStitches = itBound(k)) } }
                        }
                    )
                    Spacer(Modifier.height(8.dp))
                    ReadOnlyLine(S.size.widthStitches, estimateWidth(s, st.aspect).toString())
                }
                SizePick.BY_DPI -> {
                    NumberField(
                        label = S.size.density,
                        value = s.presetDensity.toString(),
                        onValue = { v -> v.toFloatOrNull()?.let { k -> vm.updateSize { s0 -> s0.copy(presetDensity = k) } }
                        }
                    )
                    Spacer(Modifier.height(8.dp))
                    ReadOnlyLine(S.size.byDpi, calcPhysicalSizeText(S, s, st.aspect))
                }
            }

            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Checkbox(checked = s.keepAspect, onCheckedChange = { c ->
                    vm.updateSize { it.copy(keepAspect = c) }
                })
                Text(S.size.keepAspect)
            }

            Spacer(Modifier.height(8.dp))
            NumberField(
                label = S.size.pagesTarget,
                value = s.pagesTarget.toString(),
                onValue = { v -> v.toIntOrNull()?.let { k -> vm.updateSize { s0 -> s0.copy(pagesTarget = k.coerceAtLeast(1)) } }
                }
            )

            Spacer(Modifier.height(12.dp))
            Text(physicalSummaryLine(S, s, st.aspect), style = MaterialTheme.typography.bodyMedium)
        }

        // Undo/Redo + Apply
        val canUndo by vm.canUndo.collectAsState()
        val canRedo by vm.canRedo.collectAsState()
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(enabled = canUndo, onClick = { vm.undo() }) { Text(LocalStrings.current.common.undo) }
            OutlinedButton(enabled = canRedo, onClick = { vm.redo() }) { Text(LocalStrings.current.common.redo) }
            Spacer(Modifier.weight(1f))
            ApplyBar(enabled = st.sourceImage != null && !st.isBusy) { vm.applySize() }
        }
    }
}

private fun itBound(it: Int) = it.coerceIn(1, 50000)

@Composable private fun SegButton(text: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(selected = selected, onClick = onClick, label = { Text(text) })
}

@Composable private fun NumberField(label: String, value: String, onValue: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValue,
        label = { Text(label) },
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable private fun ReadOnlyLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
}

private fun densityLabel(S: Strings, d: Float, unit: UnitMode) =
    if (unit == UnitMode.ST_PER_INCH) S.size.densityLabelInch(d) else S.size.densityLabelCm(d)

private fun estimateHeight(s: SizeState, aspect: Float): Int =
    if (!s.keepAspect) s.heightStitches else (s.widthStitches / aspect).toInt().coerceAtLeast(1)

private fun estimateWidth(s: SizeState, aspect: Float): Int =
    if (!s.keepAspect) s.widthStitches else (s.heightStitches * aspect).toInt().coerceAtLeast(1)

private fun physicalSummaryLine(S: Strings, s: SizeState, aspect: Float): String {
    val w = if (s.pick == SizePick.BY_HEIGHT) estimateWidth(s, aspect) else s.widthStitches
    val h = if (s.pick == SizePick.BY_WIDTH) estimateHeight(s, aspect) else s.heightStitches
    val wIn = w / maxOf(1f, s.presetDensity)
    val hIn = h / maxOf(1f, s.presetDensity)
    return S.size.physicalSummary(s.presetDensity, wIn, wIn * 2.54f, hIn, hIn * 2.54f)
}

private fun calcPhysicalSizeText(S: Strings, s: SizeState, aspect: Float): String {
    val w = s.widthStitches
    val h = if (s.keepAspect) (w / aspect) else s.heightStitches
    val wIn = w.toFloat() / maxOf(1f, s.presetDensity)
    val hIn = h.toFloat() / maxOf(1f, s.presetDensity)
    return S.size.physicalSizeText(wIn, hIn)
}