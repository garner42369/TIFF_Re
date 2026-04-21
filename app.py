import streamlit as st
import os
import tempfile
import logging
import math
import uuid
import pandas as pd
import numpy as np

# 使用 rasterio 替代原生 GDAL，自带预编译 C++ 库，云端部署 100% 成功
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    rasterio_available = True
except ImportError:
    rasterio_available = False

# Setup wide layout
st.set_page_config(page_title="栅格数据批处理与多值提取工具", layout="wide")

# Custom Streamlit Logging Handler
class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_placeholder):
        super().__init__()
        self.log_placeholder = log_placeholder
        self.logs = []

    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
        # Keep only last 100 logs to prevent UI lag
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
        self.log_placeholder.text_area("运行日志", "\n".join(self.logs), height=200, key=f"log_{len(self.logs)}")

@st.cache_data(show_spinner=False, max_entries=100)
def get_raster_info_cached(file_path, file_size):
    """
    带缓存机制的栅格元数据提取 (使用 Rasterio 重写)
    """
    try:
        with rasterio.open(file_path) as src:
            proj = src.crs.to_wkt() if src.crs else ""
            crs_name = src.crs.name if src.crs else "Unknown"
            epsg = src.crs.to_epsg() if src.crs else "Unknown"
            
            # rasterio.transform: Affine(a, b, c, d, e, f)
            # a=res_x, c=min_x, e=res_y, f=max_y
            gt = src.transform
            res_x = gt.a
            res_y = gt.e
            minx = src.bounds.left
            maxy = src.bounds.top
            maxx = src.bounds.right
            miny = src.bounds.bottom
            
            cols = src.width
            rows = src.height
            nodata = src.nodata
            
            return {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "crs_wkt": proj,
                "crs_name": f"{crs_name} (EPSG:{epsg})",
                "res_x": res_x,
                "res_y": res_y,
                "min_x": minx,
                "max_x": maxx,
                "min_y": miny,
                "max_y": maxy,
                "cols": cols,
                "rows": rows,
                "nodata": nodata
            }
    except Exception as e:
        return None

def get_bbox_in_target_crs(info, target_wkt):
    """使用 rasterio 原生的 transform_bounds 处理极地投影弧度变形问题"""
    if info['crs_wkt'] == target_wkt:
        return info['min_x'], info['min_y'], info['max_x'], info['max_y']
    
    source_crs = CRS.from_wkt(info['crs_wkt'])
    target_crs = CRS.from_wkt(target_wkt)
    
    try:
        # transform_bounds 内置 densify_pts 参数，能准确获取变形后的包围盒极限值
        left, bottom, right, top = transform_bounds(
            source_crs, target_crs,
            info['min_x'], info['min_y'], info['max_x'], info['max_y']
        )
        return left, bottom, right, top
    except Exception:
        raise ValueError(f"无法将 {info['filename']} 投影转换至主栅格坐标系。")

def init_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.temp_dir = tempfile.mkdtemp(prefix=f"raster_app_{st.session_state.session_id}_")
        st.session_state.raster_infos = []
        st.session_state.processed_files_hash = ""

def main():
    st.title("🌐 栅格数据批处理与多值提取工具 (云端免编译版)")
    st.markdown("基于 **Rasterio (纯 Python 预编译版)** 重构，完美避开 GDAL C++ 编译噩梦。采用内存分块读取 (Chunking) 彻底解决多用户并发时的内存溢出问题。")
    
    if not rasterio_available:
        st.error("严重错误: 系统未检测到 rasterio 库。请执行 `pip install rasterio`。")
        return
        
    init_session()
    
    # 1. File Upload
    uploaded_files = st.file_uploader("选择并上传栅格文件 (支持多选，如 GeoTIFF)", accept_multiple_files=True, type=['tif', 'tiff'])
    
    if not uploaded_files or len(uploaded_files) < 2:
        st.info("请上传至少两个栅格文件以进行多值提取。")
        return

    current_hash = "|".join([f"{f.name}_{f.size}" for f in uploaded_files])
    
    if current_hash != st.session_state.processed_files_hash:
        raster_infos = []
        with st.spinner("正在安全隔离并读取栅格元数据 (Rasterio 引擎)..."):
            for uf in uploaded_files:
                temp_path = os.path.join(st.session_state.temp_dir, uf.name)
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) != uf.size:
                    with open(temp_path, "wb") as f:
                        f.write(uf.getbuffer())
                
                info = get_raster_info_cached(temp_path, uf.size)
                if info:
                    raster_infos.append(info)
                else:
                    st.error(f"文件 {uf.name} 无法被读取为有效栅格，已跳过。")
                    
        st.session_state.raster_infos = raster_infos
        st.session_state.processed_files_hash = current_hash
        
    raster_infos = st.session_state.raster_infos
    if len(raster_infos) < 2:
        st.error("有效的栅格文件不足，无法继续。")
        return

    # 2. Master Raster Selection
    st.subheader("1. 栅格数据概览与主栅格选择")
    df_info = pd.DataFrame(raster_infos)
    display_df = df_info[['filename', 'crs_name', 'res_x', 'res_y', 'min_x', 'min_y', 'max_x', 'max_y', 'cols', 'rows']].copy()
    display_df.columns = ['文件名', '坐标系', 'X分辨率', 'Y分辨率', '最小X', '最小Y', '最大X', '最大Y', '列数', '行数']
    st.dataframe(display_df, use_container_width=True)
    
    master_filename = st.selectbox("选择主栅格 (状态已被缓存，切换不卡顿):", options=[r['filename'] for r in raster_infos])
    master_info = next(r for r in raster_infos if r['filename'] == master_filename)
    st.info(f"**主栅格信息**: {master_info['crs_name']} | 分辨率: ({master_info['res_x']:.6f}, {master_info['res_y']:.6f})")

    # Processing trigger
    col1, col2 = st.columns([3, 1])
    with col1:
        start_btn = st.button("开始处理 (内存分块防 OOM 模式)", type="primary", use_container_width=True)
    with col2:
        chunk_size_option = st.selectbox("分块读取速度 (内存开销)", ["10,000 行 (推荐)", "2,000 行 (极低内存)", "50,000 行 (较快)", "100,000 行 (极快但吃内存)"], index=0)
        chunk_mapping = {"2,000 行 (极低内存)": 2000, "10,000 行 (推荐)": 10000, "50,000 行 (较快)": 50000, "100,000 行 (极快但吃内存)": 100000}
        selected_chunk_size = chunk_mapping[chunk_size_option]

    if start_btn:
        log_container = st.empty()
        logger = logging.getLogger("RasterProcessing")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        
        log_handler = StreamlitLogHandler(log_container)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(log_handler)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 3. Calculate Common Extent in Master CRS
            logger.info("步骤 1/4: 计算共有空间范围...")
            status_text.text("步骤 1/4: 计算共有空间范围")
            
            common_min_x, common_min_y = -float('inf'), -float('inf')
            common_max_x, common_max_y = float('inf'), float('inf')
            
            for info in raster_infos:
                minx, miny, maxx, maxy = get_bbox_in_target_crs(info, master_info['crs_wkt'])
                common_min_x, common_min_y = max(common_min_x, minx), max(common_min_y, miny)
                common_max_x, common_max_y = min(common_max_x, maxx), min(common_max_y, maxy)
                
            if common_min_x >= common_max_x or common_min_y >= common_max_y:
                logger.error("栅格数据不相交，处理终止。")
                st.error("错误: 所选栅格数据在空间上不相交。")
                return
                
            # Snap to Master Grid
            res_x, res_y = master_info['res_x'], master_info['res_y']
            m_min_x, m_max_y = master_info['min_x'], master_info['max_y']
            
            new_min_x = m_min_x + math.ceil((common_min_x - m_min_x) / res_x) * res_x
            new_max_x = m_min_x + math.floor((common_max_x - m_min_x) / res_x) * res_x
            new_max_y = m_max_y + math.ceil((common_max_y - m_max_y) / res_y) * res_y
            new_min_y = m_max_y + math.floor((common_min_y - m_max_y) / res_y) * res_y
            
            if new_min_x >= new_max_x or new_min_y >= new_max_y:
                 st.error("对齐失败: 共有范围小于一个像元大小。")
                 return
                 
            # 4. Projection & Resampling (Rasterio reproject)
            status_text.text("步骤 2/4: 执行投影转换与对齐 (Rasterio Engine)")
            logger.info("步骤 2/4: 执行投影转换与对齐 (直接落盘，不占用系统内存)...")
            aligned_files = []
            
            # Calculate output shape and transform
            out_transform = Affine(res_x, 0.0, new_min_x, 0.0, res_y, new_max_y)
            out_width = int(round(abs((new_max_x - new_min_x) / res_x)))
            out_height = int(round(abs((new_min_y - new_max_y) / res_y)))
            
            dst_crs = CRS.from_wkt(master_info['crs_wkt'])
            
            for i, info in enumerate(raster_infos):
                progress_bar.progress(10 + int((i / len(raster_infos)) * 30))
                out_path = os.path.join(st.session_state.temp_dir, f"aligned_{info['filename']}")
                
                with rasterio.open(info['file_path']) as src:
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': dst_crs,
                        'transform': out_transform,
                        'width': out_width,
                        'height': out_height,
                        'compress': 'lzw',
                        'tiled': True
                    })
                    
                    with rasterio.open(out_path, 'w', **kwargs) as dst:
                        for j in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, j),
                                destination=rasterio.band(dst, j),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=out_transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.bilinear
                            )
                            
                aligned_files.append((info['filename'], out_path))
                
            # 5. Multi-value Extraction via Chunking (防爆核心)
            status_text.text("步骤 3/4: 执行分块多值提取 (防 OOM 机制)")
            logger.info("步骤 3/4: 开始分块读取与提取，启动内存防爆 (Chunking) 机制...")
            
            # Open all aligned datasets with rasterio
            datasets = [rasterio.open(path) for _, path in aligned_files]
            
            rows, cols = out_height, out_width
            logger.info(f"对齐后共有范围网格: {cols} 列 x {rows} 行, 总像元 {cols*rows}。")
            
            csv_path = os.path.join(st.session_state.temp_dir, "multi_value_extraction_result.csv")
            raster_cols = [fname for fname, _ in aligned_files]
            header = ['X', 'Y'] + raster_cols
            
            pd.DataFrame(columns=header).to_csv(csv_path, index=False)
            
            CHUNK_SIZE = selected_chunk_size
            logger.info(f"当前分块配置: 每次加载 {CHUNK_SIZE} 行。")
            valid_pixels_count = 0
            
            x_coords = new_min_x + res_x/2 + np.arange(cols) * res_x
            
            for yoff in range(0, rows, CHUNK_SIZE):
                ysize = min(CHUNK_SIZE, rows - yoff)
                
                y_coords = new_max_y + res_y/2 + (np.arange(ysize) + yoff) * res_y
                xv, yv = np.meshgrid(x_coords, y_coords)
                
                chunk_dict = {'X': xv.flatten(), 'Y': yv.flatten()}
                
                # rasterio Window(col_off, row_off, width, height)
                from rasterio.windows import Window
                window = Window(0, yoff, cols, ysize)
                
                for ds, (fname, _) in zip(datasets, aligned_files):
                    nodata = ds.nodata
                    # Read only the window
                    arr = ds.read(1, window=window).flatten()
                    
                    if nodata is not None:
                        if isinstance(nodata, (int, float)) and not pd.isna(nodata):
                            arr = np.where(arr == nodata, np.nan, arr)
                            
                    chunk_dict[fname] = arr
                    
                df_chunk = pd.DataFrame(chunk_dict)
                df_chunk.dropna(subset=raster_cols, how='all', inplace=True)
                
                if not df_chunk.empty:
                    df_chunk.to_csv(csv_path, mode='a', header=False, index=False)
                    valid_pixels_count += len(df_chunk)
                
                del chunk_dict, df_chunk
                
                progress = 40 + int(55 * ((yoff + ysize) / rows))
                progress_bar.progress(progress)
                
            # Close all datasets
            for ds in datasets:
                ds.close()
            
            progress_bar.progress(100)
            status_text.text("处理完成！")
            logger.info(f"提取完成！采用 Rasterio 分块 I/O，零内存溢出。累计写入有效像元: {valid_pixels_count} 条。")
            st.success(f"处理完成！共提取 {valid_pixels_count} 个有效像元。")
            
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 下载多值提取结果 (CSV)", data=f,
                    file_name="raster_extraction_results.csv", mime="text/csv", type="primary"
                )
                
            preview_df = pd.read_csv(csv_path, nrows=100)
            with st.expander("预览提取结果 (前 100 行)"):
                st.dataframe(preview_df, use_container_width=True)
                
        except Exception as e:
            logger.error(f"处理异常: {str(e)}", exc_info=True)
            st.error(f"处理失败: {str(e)}")
            
if __name__ == "__main__":
    main()