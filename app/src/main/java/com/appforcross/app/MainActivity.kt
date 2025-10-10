//Hash fcab643f8af2d95e10a4ef289f8977af
package com.appforcross.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.EditorViewModelFactory
import com.appforcross.editor.engine.CoreEngine
import com.appforcross.app.palette.AndroidPaletteRepository
import com.appforcross.editor.ui.EditorScreen
import androidx.lifecycle.ViewModelProvider
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import com.appforcross.i18n.AndroidStrings
import com.appforcross.i18n.LocalStrings

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Surface(color = MaterialTheme.colorScheme.background) {
                val vm: EditorViewModel =
                    ViewModelProvider(
                        this@MainActivity,
                        EditorViewModelFactory(CoreEngine())
                    ).get(EditorViewModel::class.java)
                vm.setPaletteRepository(AndroidPaletteRepository(applicationContext))
                val ctx = LocalContext.current
                val strings = remember(ctx) { AndroidStrings(ctx.resources) }
                CompositionLocalProvider(LocalStrings provides strings) {
                    EditorScreen(vm)
                }
            }
        }
    }
}
