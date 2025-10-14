package com.appforcross.editor.photo.hq

import android.util.Log

/** Мини‑логгер для PhotoHQ Orchestrator: стабильные теги под греп. */
internal object HQLog {

    fun pre(msg: String) = Log.i("PhotoHQ.Pre", msg)

    fun mask(msg: String) = Log.d("PhotoHQ.Mask", msg)

    fun scale(msg: String)= Log.d("PhotoHQ.Scale", msg)

    fun auto(msg: String) = Log.d("PhotoHQ.AutoTune", msg)

    fun kmeans(msg: String)= Log.d("PhotoHQ.KMeans", msg)

    fun map(msg: String) = Log.d("PhotoHQ.Map", msg)

    fun s(msg: String) = Log.d("PhotoHQ.S", msg)

    fun out(msg: String) = Log.i("PhotoHQ.Out", msg)

}