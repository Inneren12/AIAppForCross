package com.appforcross.editor

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.appforcross.editor.engine.EditorEngine

class EditorViewModelFactory(private val engine: EditorEngine) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(EditorViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return EditorViewModel(engine) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
