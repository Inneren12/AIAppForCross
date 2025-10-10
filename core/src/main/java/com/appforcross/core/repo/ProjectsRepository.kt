package com.appforcross.core.repo

import com.appforcross.core.image.Project
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.util.UUID

class ProjectsRepository {

    private val _projects = MutableStateFlow<List<Project>>(emptyList())
    val projects: StateFlow<List<Project>> = _projects.asStateFlow()

    fun observe(): StateFlow<List<Project>> = projects
    fun list(): List<Project> = projects.value
    /** Загрузить проект по id */
    fun load(id: String): Project? =
        _projects.value.firstOrNull { it.id == id }

    /** Создать новый проект */
    fun create(name: String): Project {
        val project = Project(
            id = UUID.randomUUID().toString(),
            params = com.appforcross.core.image.ProjectParams(name = name)
        )
        _projects.update { list: List<Project> ->
            list + project
        }
        return project
    }

    /** Сохранить/обновить проект */
    fun save(project: Project): Project {
        val updated: Project = project.copy(updatedAt = System.currentTimeMillis())
        _projects.update { list: List<Project> ->
            val idx = list.indexOfFirst { p: Project -> p.id == updated.id }
            if (idx >= 0) {
                val m = list.toMutableList()
                m[idx] = updated
                m
            } else {
                list + updated
            }
        }
        return updated
    }

    /** Сделать копию проекта */
    fun copy(id: String): Project? {
        val orig = load(id) ?: return null
        val clone = orig.copy(
            id = UUID.randomUUID().toString(),
            params = orig.params.copy(name = orig.params.name + " (копия)"),
            updatedAt = System.currentTimeMillis()
        )
        _projects.update { list: List<Project> -> list + clone }
        return clone
    }

    /** Удалить проект */
    fun delete(id: String) {
        _projects.update { list: List<Project> ->
            list.filter { it.id != id }
        }
    }

    /** Пометить проект как обновлённый по id */
    fun touch(id: String) {
        _projects.update { list: List<Project> ->
            list.map { p: Project ->
                if (p.id == id) p.copy(updatedAt = System.currentTimeMillis()) else p
            }
        }
    }
}
