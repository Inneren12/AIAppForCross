//Hash 66e4116707ddae9e8a0e7527daa22ed2
package com.appforcross.editor.ui.tabs

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.model.*
import com.appforcross.editor.ui.components.ApplyBar
import com.appforcross.editor.ui.components.Section
import com.appforcross.i18n.LocalStrings
import androidx.compose.ui.Alignment
import androidx.compose.ui.res.stringResource
import com.appforcross.app.R

@Composable
fun OptionsTab(vm: EditorViewModel) {
    val st by vm.state.collectAsState()
    val o = st.options
    val S = LocalStrings.current

    Column(Modifier.fillMaxSize()) {

        Section(S.options.paletteTitle) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Seg(text = S.options.dmcAsIs, selected = o.useDmcAsIs) {
                    vm.updateOptions { it.copy(useDmcAsIs = true) }
                }
                Seg(text = S.options.adaptiveToDmc, selected = !o.useDmcAsIs) {
                    vm.updateOptions { it.copy(useDmcAsIs = false) }
                }
            }

            Spacer(Modifier.height(8.dp))
            Text(S.options.mergeDeltaE(o.mergeDeltaE))
            Slider(
                value = o.mergeDeltaE,
                onValueChange = { v -> vm.updateOptions { it.copy(mergeDeltaE = v) } },
                valueRange = 0f..20f
            )
        }

        Section(S.options.ditherTitle) {
            Text(S.options.ditherStrength(o.fsStrengthPct))
            Slider(
                value = o.fsStrengthPct.toFloat(),
                onValueChange = { v -> vm.updateOptions { it.copy(fsStrengthPct = v.toInt()) } },
                valueRange = 0f..100f
            )
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Checkbox(checked = o.cleanSingles, onCheckedChange = { c ->
                    vm.updateOptions { it.copy(cleanSingles = c) }
                })
                Text(S.options.cleanSingles)
            }

            // Edge‑enhance before quantization (VM‑тумблер; без изменения EditorState)
            val edge by vm.edgeEnhanceEnabled.collectAsState()
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Checkbox(checked = edge, onCheckedChange = { vm.setEdgeEnhanceEnabled(it) })
                Text(stringResource(id = R.string.options_edge_enhance))
            }
        }

        Section(S.options.resampleTitle) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Seg(text = S.options.avgPerCell, selected = o.resampling == ResamplingMode.AVERAGE_PER_CELL) {
                    vm.updateOptions { it.copy(resampling = ResamplingMode.AVERAGE_PER_CELL) }
                }
                Seg(text = S.options.simpleFast, selected = o.resampling == ResamplingMode.SIMPLE_FAST) {
                    vm.updateOptions { it.copy(resampling = ResamplingMode.SIMPLE_FAST) }
                }
            }

            Spacer(Modifier.height(8.dp))
            Text(S.options.preblurSigma(o.preBlurSigmaPx))
            Slider(
                value = o.preBlurSigmaPx,
                onValueChange = { v -> vm.updateOptions { it.copy(preBlurSigmaPx = v) } },
                valueRange = 0f..5f
            )
            Text(S.options.recommendation)
        }

        Spacer(Modifier.weight(1f))
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
            ApplyBar(enabled = st.previewImage != null && !st.isBusy) { vm.applyOptions() }
        }
    }
}

@Composable private fun Seg(text: String, selected: Boolean, onClick: () -> Unit) {
    FilterChip(selected = selected, onClick = onClick, label = { Text(text) })
}