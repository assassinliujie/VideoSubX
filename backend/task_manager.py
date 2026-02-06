import threading
import time
import os
import shutil
import sys
from datetime import datetime
from backend.global_state import state, TaskStatus
import core.utils as utils

# 如果 core 不在 sys.path 中则添加（通常在 main.py 中处理，但这里也安全处理）
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入核心模块
from core import downloader, transcriber, splitter_nlp, splitter_meaning, summarizer, translator, subtitle_splitter, subtitle_generator, subtitle_burner

class TaskManager:
    def __init__(self):
        self.workflow_thread = None
        self.stop_flag = threading.Event()

    def start_workflow(self, url: str):
        if self.workflow_thread and self.workflow_thread.is_alive():
            state.add_log("Workflow already running.")
            return

        # 自动归档并重置
        state.add_log("Auto-archiving previous run...")
        self.reset_workspace()

        self.stop_flag.clear()
        
        # 启动主编排线程
        self.workflow_thread = threading.Thread(target=self._workflow_runner, args=(url,))
        self.workflow_thread.daemon = True
        self.workflow_thread.start()
        state.add_log(f"Starting workflow for URL: {url}")

    def stop_workflow(self):
        self.stop_flag.set()
        state.set_status(TaskStatus.IDLE)
        state.add_log("Workflow stop requested.")

    def _workflow_runner(self, url):
        try:
            # 步骤 1: 下载 360p
            state.set_status(TaskStatus.DOWNLOADING_360P)
            state.update_task_status("download_360p", "running")
            state.add_log("Step 1: Downloading 360p video for audio extraction...")
            
            downloader.download_video_ytdlp(url, resolution='360')
            
            state.update_task_status("download_360p", "completed")
            state.add_log("360p download complete.")

            if self.stop_flag.is_set(): return

            # 步骤 2: 并行任务
            state.set_status(TaskStatus.PROCESSING)
            
            # 任务 A: 处理
            def run_processing():
                state.update_task_status("process_transcription", "running")
                try:
                    state.add_log("Task A: Starting Audio/Subtitle Processing...")
                    
                    state.add_log("Running Whisper ASR...")
                    transcriber.transcribe()
                    
                    state.add_log("Splitting sentences (NLP)...")
                    splitter_nlp.split_by_spacy()
                    
                    state.add_log("Splitting sentences (Meaning)...")
                    splitter_meaning.split_sentences_by_meaning()
                    
                    state.add_log("Summarizing...")
                    summarizer.get_summary()
                    
                    state.add_log("Translating...")
                    translator.translate_all()
                    
                    state.add_log("Splitting for subtitles...")
                    subtitle_splitter.split_for_sub_main()
                    
                    state.add_log("Aligning timestamps...")
                    subtitle_generator.align_timestamp_main()
                    
                    state.update_task_status("process_transcription", "completed")
                    state.add_log("Task A: Processing pipeline completed successfully.")
                except Exception as e:
                    state.add_log(f"Task A Failed: {str(e)}")
                    state.update_task_status("process_transcription", "error")
                    state.status = TaskStatus.ERROR
                    import traceback
                    print(traceback.format_exc())

            # 任务 B: 下载最佳质量
            def run_download_best():
                state.update_task_status("download_best", "running")
                try:
                    state.add_log("Task B: Downloading Best Quality Video...")
                    downloader.download_video_ytdlp(url, resolution='best', suffix='_best')
                    state.update_task_status("download_best", "completed")
                    state.add_log("Task B: Best quality download complete.")
                except Exception as e:
                    state.add_log(f"Task B Failed: {str(e)}")
                    state.update_task_status("download_best", "error")

            thread_a = threading.Thread(target=run_processing, name="_task_processing")
            thread_b = threading.Thread(target=run_download_best, name="_task_download_best")
            
            thread_a.start()
            thread_b.start()
            
            thread_a.join()
            thread_b.join()

            if state.status != TaskStatus.ERROR:
                state.set_status(TaskStatus.COMPLETED)
                state.add_log("Workflow All Completed.")

        except Exception as e:
            state.set_status(TaskStatus.ERROR)
            state.add_log(f"Workflow Critical Error: {str(e)}")
            import traceback
            state.add_log(traceback.format_exc())

    def start_local_workflow(self):
        """使用本地上传的视频/音频开始工作流，跳过下载步骤"""
        if self.workflow_thread and self.workflow_thread.is_alive():
            state.add_log("Workflow already running.")
            return

        # 自动归档并重置
        state.add_log("Auto-archiving previous run...")
        self.reset_workspace(skip_output_clean=True)  # 不清理 output，因为用户刚上传了文件

        self.stop_flag.clear()
        
        # 启动本地处理线程
        self.workflow_thread = threading.Thread(target=self._local_workflow_runner)
        self.workflow_thread.daemon = True
        self.workflow_thread.start()
        state.add_log("Starting local file workflow (skipping download)...")

    def _local_workflow_runner(self):
        """本地文件工作流：跳过下载，直接处理"""
        try:
            # 标记下载任务为已完成（跳过）
            state.update_task_status("download_360p", "completed")
            state.update_task_status("download_best", "completed")
            state.add_log("Download steps skipped (using local file).")

            if self.stop_flag.is_set(): return

            # 直接执行处理流程
            state.set_status(TaskStatus.PROCESSING)
            state.update_task_status("process_transcription", "running")
            
            try:
                state.add_log("Starting Audio/Subtitle Processing...")
                
                state.add_log("Running Whisper ASR...")
                transcriber.transcribe()
                
                state.add_log("Splitting sentences (NLP)...")
                splitter_nlp.split_by_spacy()
                
                state.add_log("Splitting sentences (Meaning)...")
                splitter_meaning.split_sentences_by_meaning()
                
                state.add_log("Summarizing...")
                summarizer.get_summary()
                
                state.add_log("Translating...")
                translator.translate_all()
                
                state.add_log("Splitting for subtitles...")
                subtitle_splitter.split_for_sub_main()
                
                state.add_log("Aligning timestamps...")
                subtitle_generator.align_timestamp_main()
                
                state.update_task_status("process_transcription", "completed")
                state.set_status(TaskStatus.COMPLETED)
                state.add_log("Local file workflow completed successfully.")
                
            except Exception as e:
                state.add_log(f"Processing Failed: {str(e)}")
                state.update_task_status("process_transcription", "error")
                state.set_status(TaskStatus.ERROR)
                import traceback
                state.add_log(traceback.format_exc())

        except Exception as e:
            state.set_status(TaskStatus.ERROR)
            state.add_log(f"Local Workflow Critical Error: {str(e)}")
            import traceback
            state.add_log(traceback.format_exc())

    def continue_workflow(self):
        """继续上次中断的任务，从断点处继续执行（不清理工作空间，不重新下载）"""
        if self.workflow_thread and self.workflow_thread.is_alive():
            state.add_log("Workflow already running.")
            return

        self.stop_flag.clear()
        
        # 不清理工作空间，直接从断点继续
        self.workflow_thread = threading.Thread(target=self._continue_runner)
        self.workflow_thread.daemon = True
        self.workflow_thread.start()
        state.add_log("Continuing workflow from last checkpoint...")

    def _continue_runner(self):
        """继续执行工作流，直接运行处理流程，@check_file_exists 会自动跳过已完成步骤"""
        try:
            state.set_status(TaskStatus.PROCESSING)
            state.update_task_status("process_transcription", "running")
            state.add_log("Checking completed steps and resuming...")
            
            try:
                # 直接按顺序调用，装饰器会自动跳过已有输出文件的步骤
                state.add_log("Running Whisper ASR...")
                transcriber.transcribe()
                
                state.add_log("Splitting sentences (NLP)...")
                splitter_nlp.split_by_spacy()
                
                state.add_log("Splitting sentences (Meaning)...")
                splitter_meaning.split_sentences_by_meaning()
                
                state.add_log("Summarizing...")
                summarizer.get_summary()
                
                state.add_log("Translating...")
                translator.translate_all()
                
                state.add_log("Splitting for subtitles...")
                subtitle_splitter.split_for_sub_main()
                
                state.add_log("Aligning timestamps...")
                subtitle_generator.align_timestamp_main()
                
                state.update_task_status("process_transcription", "completed")
                state.set_status(TaskStatus.COMPLETED)
                state.add_log("Continue Workflow Completed.")
                
            except Exception as e:
                state.add_log(f"Continue Failed: {str(e)}")
                state.update_task_status("process_transcription", "error")
                state.set_status(TaskStatus.ERROR)
                import traceback
                state.add_log(traceback.format_exc())

        except Exception as e:
            state.set_status(TaskStatus.ERROR)
            state.add_log(f"Continue Critical Error: {str(e)}")
            import traceback
            state.add_log(traceback.format_exc())

    def burn_video(self):
        state.update_task_status("burn_video", "running")
        state.add_log("Starting Subtitle Burning...")
        
        def run_burn():
            try:
                subtitle_burner.burn_subtitle_to_video()
                state.update_task_status("burn_video", "completed")
                state.add_log("Burning completed successfully.")
            except Exception as e:
                state.update_task_status("burn_video", "error")
                state.add_log(f"Burning failed: {str(e)}")

        threading.Thread(target=run_burn, daemon=True).start()

    def reset_workspace(self, skip_output_clean=False):
        """
        归档当前执行并清理 output 文件夹。
        skip_output_clean: 如果为 True，只归档字幕文件但不清理 output 目录（用于本地上传场景）
        """
        output_dir = "output"
        archive_dir = "archives"
        
        os.makedirs(archive_dir, exist_ok=True)
        
        # 检查 .ass 文件
        ass_file = os.path.join(output_dir, "src_trans.ass")
        if os.path.exists(ass_file):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_name = f"{timestamp}.ass"
            archive_path = os.path.join(archive_dir, new_name)
            shutil.move(ass_file, archive_path)
            state.add_log(f"Archived subtitle to {archive_path}")
        
        # 重置状态
        state.reset()
        
        # 如果跳过清理（本地上传场景），直接返回
        if skip_output_clean:
            state.add_log("Skipping output cleanup (local file mode).")
            return
            
        # 清理 output 目录
        try:
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    state.add_log(f"Failed to delete {file_path}. Reason: {e}")
            state.add_log("Output directory cleaned.")
        except Exception as e:
             state.add_log(f"Error cleaning output directory: {e}")

task_manager = TaskManager()
