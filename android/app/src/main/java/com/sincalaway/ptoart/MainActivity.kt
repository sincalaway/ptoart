package com.sincalaway.ptoart

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.sincalaway.ptoart.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private lateinit var ui: ActivityMainBinding
    private var photo: Bitmap? = null
    private var result: Bitmap? = null
    private var cartoonizer: Cartoonizer? = null
    private var styleModel: Pair<File, StyleModel>? = null
    private val modelFile by lazy { File(filesDir, "photo2cartoon_weights.onnx") }
    private val upscalerFile by lazy { File(filesDir, "upscaler.onnx") }
    private var upscaler: Upscaler? = null
    private val styleDir by lazy { File(filesDir, "styles").apply { mkdirs() } }

    /** What the user can produce: face modes need photo2cartoon, styles need their own file. */
    private sealed class Mode(val label: String) {
        class Face(label: String) : Mode(label)
        class Photo(label: String) : Mode(label)
        class Style(label: String, val file: File) : Mode(label)
        override fun toString() = label
    }
    private var modes: List<Mode> = emptyList()
    private val currentMode get() = modes.getOrNull(ui.mode.selectedItemPosition)

    private val styleNames = mapOf(
        "hayao" to "宫崎骏", "shinkai" to "新海诚", "paprika" to "今敏 Paprika",
        "celeba_distill" to "CelebA 蒸馏", "face_paint_512_v1" to "Face Paint v1",
        "face_paint_512_v2" to "Face Paint v2",
    )

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isEmpty()) return@registerForActivityResult
        lifecycleScope.launch {
            busy(true, "正在导入模型…")
            val messages = withContext(Dispatchers.IO) {
                uris.map { uri ->
                    val name = displayName(uri) ?: "model.onnx"
                    runCatching {
                        val tmp = File(filesDir, "model.tmp")
                        contentResolver.openInputStream(uri)!!.use { i -> tmp.outputStream().use { i.copyTo(it) } }
                        when (ModelInspector.kindOf(tmp)) { // validates before replacing anything
                            ModelKind.PHOTO2CARTOON -> {
                                cartoonizer?.close(); cartoonizer = null
                                check(tmp.renameTo(modelFile)) { "无法保存模型" }
                                "已导入卡通头像模型"
                            }
                            ModelKind.UPSCALER -> {
                                cartoonizer?.upscaler = null
                                upscaler?.close(); upscaler = null
                                check(tmp.renameTo(upscalerFile)) { "无法保存模型" }
                                "已导入超分辨率模型"
                            }
                            ModelKind.STYLE -> {
                                val stem = name.substringBeforeLast('.').ifBlank { "style" }
                                val target = File(styleDir, "$stem.onnx")
                                if (styleModel?.first == target) { styleModel?.second?.close(); styleModel = null }
                                check(tmp.renameTo(target)) { "无法保存模型" }
                                "已导入风格：${styleNames[stem] ?: stem}"
                            }
                        }
                    }.getOrElse { "$name 无效：${it.message}" }
                }
            }
            busy(false)
            toast(messages.joinToString("\n"))
            refreshModelState()
        }
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri ?: return@registerForActivityResult
        lifecycleScope.launch {
            val bmp = withContext(Dispatchers.IO) { runCatching { decode(uri) }.getOrNull() }
            if (bmp == null) { toast("无法读取这张图片"); return@launch }
            photo = bmp; result = null
            ui.image.setImageBitmap(bmp)
            ui.save.isEnabled = false
            refreshModelState()
        }
    }

    private val saveResult = registerForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        val bmp = result ?: return@registerForActivityResult
        uri ?: return@registerForActivityResult
        lifecycleScope.launch(Dispatchers.IO) {
            val ok = runCatching {
                contentResolver.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }.isSuccess
            withContext(Dispatchers.Main) { toast(if (ok) "已保存" else "保存失败") }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)
        ui.importModel.setOnClickListener { pickModel.launch(arrayOf("*/*")) }
        ui.mode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) = refreshModelState(rebuild = false)
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        ui.pick.setOnClickListener { pickPhoto.launch("image/*") }
        ui.convert.setOnClickListener { convert() }
        ui.save.setOnClickListener { saveResult.launch("ptoart-${System.currentTimeMillis()}.png") }
        ui.strength.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                ui.strengthLabel.text = getString(R.string.strength_fmt, p)
            }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        ui.strengthLabel.text = getString(R.string.strength_fmt, ui.strength.progress)
        refreshModelState()
    }

    private fun refreshModelState(rebuild: Boolean = true) {
        if (rebuild) {
            val previous = currentMode?.label
            modes = buildList {
                if (modelFile.isFile) {
                    add(Mode.Face("卡通头像（最大的一张脸）"))
                    add(Mode.Photo("整张照片：所有人脸卡通化，贴回原图"))
                }
                styleDir.listFiles { f -> f.extension == "onnx" }?.sortedBy { it.name }?.forEach {
                    add(Mode.Style("整图风格：${styleNames[it.nameWithoutExtension] ?: it.nameWithoutExtension}", it))
                }
            }
            ui.mode.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, modes)
            modes.indexOfFirst { it.label == previous }.takeIf { it >= 0 }?.let { ui.mode.setSelection(it) }
        }
        val styles = modes.count { it is Mode.Style }
        ui.modelState.text = when {
            modes.isEmpty() -> getString(R.string.model_missing)
            else -> getString(R.string.model_ready_fmt, if (modelFile.isFile) "已就绪" else "未导入", styles,
                if (upscalerFile.isFile) "已就绪" else "未导入")
        }
        ui.mode.visibility = if (modes.isEmpty()) View.GONE else View.VISIBLE
        ui.transparent.visibility = if (currentMode is Mode.Face) View.VISIBLE else View.GONE
        ui.superRes.visibility = if (upscalerFile.isFile && currentMode !is Mode.Style) View.VISIBLE else View.GONE
        ui.convert.isEnabled = currentMode != null && photo != null
    }

    private fun convert() {
        val src = photo ?: return
        val mode = currentMode ?: return
        val strength = ui.strength.progress / 100f
        val transparent = ui.transparent.isChecked
        val superRes = ui.superRes.isChecked && upscalerFile.isFile
        lifecycleScope.launch {
            busy(true, "正在转换…")
            val out = withContext(Dispatchers.Default) {
                runCatching {
                    if (mode !is Mode.Style) {
                        faceModel().upscaler = if (superRes) {
                            upscaler ?: Upscaler(upscalerFile).also { upscaler = it }
                        } else null
                    }
                    when (mode) {
                        is Mode.Face -> faceModel().convertFace(src, strength, if (transparent) null else Color.WHITE)
                        is Mode.Photo -> faceModel().convertPhoto(src, strength)
                        is Mode.Style -> {
                            val current = styleModel
                            val model = if (current?.first == mode.file) current.second else {
                                current?.second?.close()
                                StyleModel(mode.file).also { styleModel = mode.file to it }
                            }
                            model.stylize(src, strength)
                        }
                    }
                }
            }
            busy(false)
            out.onSuccess {
                result = it
                ui.image.setImageBitmap(it)
                ui.save.isEnabled = true
            }.onFailure { toast(it.message ?: "转换失败") }
        }
    }

    private fun faceModel() = cartoonizer ?: Cartoonizer(modelFile).also { cartoonizer = it }

    private fun displayName(uri: Uri): String? =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        }

    private fun decode(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 1600) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }!!
        val orientation = contentResolver.openInputStream(uri)!!.use {
            ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            else -> return bmp
        }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun busy(on: Boolean, msg: String = "") {
        ui.progress.visibility = if (on) View.VISIBLE else View.GONE
        ui.status.text = msg
        listOf(ui.importModel, ui.pick, ui.convert, ui.mode).forEach { it.isEnabled = !on }
        if (!on) refreshModelState(rebuild = false)
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        cartoonizer?.close()
        styleModel?.second?.close()
        upscaler?.close()
        super.onDestroy()
    }
}
