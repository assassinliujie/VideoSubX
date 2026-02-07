import fnmatch
import os
import shutil
import sys
import threading
from datetime import datetime

from backend.global_state import state, TaskStatus

# Ensure project root is on sys.path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core import (
    downloader,
    transcriber,
    splitter_nlp,
    splitter_meaning,
    summarizer,
    translator,
    subtitle_splitter,
    subtitle_generator,
    subtitle_burner,
)


class TaskManager:
    def __init__(self):
        self.workflow_thread = None
        self.worker_threads = []
        self.stop_flag = threading.Event()
        self.local_video_filename = None
        self.local_video_source_path = None

    def set_local_video(self, filename: str, source_path: str = None):
        self.local_video_filename = os.path.basename(filename)
        self.local_video_source_path = source_path

    def clear_local_video(self):
        self.local_video_filename = None
        self.local_video_source_path = None

    def get_local_video_path(self):
        if not self.local_video_filename:
            return None
        path = os.path.join("output", self.local_video_filename)
        return path if os.path.exists(path) else None

    def _force_stop_threads(self):
        import ctypes

        def _raise_system_exit(thread_obj):
            if not thread_obj or not thread_obj.is_alive() or thread_obj.ident is None:
                return False
            tid = ctypes.c_ulong(thread_obj.ident)
            result = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
            if result == 0:
                return False
            if result > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                return False
            return True

        targets = []
        if self.workflow_thread and self.workflow_thread.is_alive():
            targets.append(self.workflow_thread)
        targets.extend([t for t in self.worker_threads if t and t.is_alive()])

        killed = 0
        for thread_obj in targets:
            if _raise_system_exit(thread_obj):
                killed += 1

        self.worker_threads = []
        return killed

    def _cleanup_download_temp_files(self):
        output_dir = "output"
        if not os.path.exists(output_dir):
            return

        removed = 0
        for root, _, files in os.walk(output_dir):
            for filename in files:
                if fnmatch.fnmatch(filename, "*.part") or fnmatch.fnmatch(filename, "*.ytdl"):
                    file_path = os.path.join(root, filename)
                    try:
                        os.remove(file_path)
                        removed += 1
                    except Exception as e:
                        state.add_log(f"Failed to remove temp file {file_path}: {e}")

        if removed > 0:
            state.add_log(f"Removed {removed} temporary download files (*.part/*.ytdl).")

    def start_workflow(self, url: str):
        if self.workflow_thread and self.workflow_thread.is_alive():
            state.add_log("Workflow already running.")
            return

        state.add_log("Auto-archiving previous run...")
        self.reset_workspace()

        self.stop_flag.clear()

        self.workflow_thread = threading.Thread(target=self._workflow_runner, args=(url,))
        self.workflow_thread.daemon = True
        self.workflow_thread.start()
        state.add_log(f"Starting workflow for URL: {url}")

    def stop_workflow(self):
        self.stop_flag.set()
        killed = self._force_stop_threads()
        self._cleanup_download_temp_files()

        for task in state.tasks.values():
            if task["status"] == "running":
                task["status"] = "stopped"

        state.set_status(TaskStatus.IDLE)
        if killed > 0:
            state.add_log("Workflow force stop requested.")
        else:
            state.add_log("Stop requested. Waiting for blocking step to exit.")

    def _workflow_runner(self, url):
        try:
            # Step 1: download low-res file for ASR
            state.set_status(TaskStatus.DOWNLOADING_360P)
            state.update_task_status("download_360p", "running")
            state.add_log("Step 1: Downloading 360p video for audio extraction...")

            downloader.download_video_ytdlp(url, resolution="360")

            state.update_task_status("download_360p", "completed")
            state.add_log("360p download complete.")

            if self.stop_flag.is_set():
                return

            # Step 2: process + best download in parallel
            state.set_status(TaskStatus.PROCESSING)

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
                    state.set_status(TaskStatus.ERROR)
                    import traceback

                    state.add_log(traceback.format_exc())

            def run_download_best():
                state.update_task_status("download_best", "running")
                try:
                    state.add_log("Task B: Downloading Best Quality Video...")
                    downloader.download_video_ytdlp(url, resolution="best", suffix="_best")
                    state.update_task_status("download_best", "completed")
                    state.add_log("Task B: Best quality download complete.")
                except Exception as e:
                    state.add_log(f"Task B Failed: {str(e)}")
                    state.update_task_status("download_best", "error")

            thread_a = threading.Thread(target=run_processing, name="_task_processing")
            thread_b = threading.Thread(target=run_download_best, name="_task_download_best")
            self.worker_threads = [thread_a, thread_b]

            thread_a.start()
            thread_b.start()

            thread_a.join()
            thread_b.join()

            if self.stop_flag.is_set():
                state.add_log("Workflow stopped.")
                return

            if state.status != TaskStatus.ERROR:
                state.set_status(TaskStatus.COMPLETED)
                state.add_log("Workflow All Completed.")

        except Exception as e:
            state.set_status(TaskStatus.ERROR)
            state.add_log(f"Workflow Critical Error: {str(e)}")
            import traceback

            state.add_log(traceback.format_exc())
        finally:
            self.worker_threads = []

    def start_local_workflow(self, local_video_filename=None, local_video_source_path=None):
        """Use uploaded local video/audio file and skip download steps."""
        if self.workflow_thread and self.workflow_thread.is_alive():
            state.add_log("Workflow already running.")
            return

        if local_video_filename:
            self.local_video_filename = os.path.basename(local_video_filename)
        if local_video_source_path:
            self.local_video_source_path = local_video_source_path

        if not self.local_video_filename:
            state.add_log("Local workflow aborted: missing local video filename.")
            state.set_status(TaskStatus.ERROR)
            return

        source_path = self.local_video_source_path

        state.add_log("Auto-archiving previous run...")
        self.reset_workspace()

        if not source_path or not os.path.exists(source_path):
            state.add_log("Local workflow aborted: cached local input file not found.")
            state.set_status(TaskStatus.ERROR)
            return

        output_target = os.path.join("output", self.local_video_filename)
        try:
            shutil.copy2(source_path, output_target)
            state.add_log(f"Prepared local input file: {self.local_video_filename}")
        except Exception as e:
            state.add_log(f"Local workflow aborted: failed to prepare input file. Reason: {e}")
            state.set_status(TaskStatus.ERROR)
            return

        self.stop_flag.clear()

        self.workflow_thread = threading.Thread(target=self._local_workflow_runner)
        self.workflow_thread.daemon = True
        self.workflow_thread.start()
        state.add_log("Starting local file workflow (skipping download)...")

    def _local_workflow_runner(self):
        """Local file workflow: skip download and run processing directly."""
        try:
            state.update_task_status("download_360p", "completed")
            state.update_task_status("download_best", "completed")
            state.add_log("Download steps skipped (using local file).")

            if self.stop_flag.is_set():
                return

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
        """Resume workflow from previous checkpoint without cleaning workspace."""
        if self.workflow_thread and self.workflow_thread.is_alive():
            state.add_log("Workflow already running.")
            return

        self.stop_flag.clear()

        self.workflow_thread = threading.Thread(target=self._continue_runner)
        self.workflow_thread.daemon = True
        self.workflow_thread.start()
        state.add_log("Continuing workflow from last checkpoint...")

    def _continue_runner(self):
        """Resume workflow; check_file_exists decorators skip completed steps."""
        try:
            state.set_status(TaskStatus.PROCESSING)
            state.update_task_status("process_transcription", "running")
            state.add_log("Checking completed steps and resuming...")

            try:
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

    def reset_workspace(self, preserve_files=None):
        """
        Archive subtitle artifacts and clean output directory.
        preserve_files: file names in output directory that should be kept.
        """
        output_dir = "output"
        archive_dir = "archives"
        preserve_set = {os.path.basename(f) for f in (preserve_files or []) if f}

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)

        ass_file = os.path.join(output_dir, "src_trans.ass")
        if os.path.exists(ass_file):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = os.path.join(archive_dir, f"{timestamp}.ass")
            shutil.move(ass_file, archive_path)
            state.add_log(f"Archived subtitle to {archive_path}")

        state.reset()

        try:
            for filename in os.listdir(output_dir):
                if filename in preserve_set:
                    continue

                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    state.add_log(f"Failed to delete {file_path}. Reason: {e}")

            state.add_log("Output directory cleaned.")
            if preserve_set:
                state.add_log(f"Preserved file(s): {', '.join(sorted(preserve_set))}")
        except Exception as e:
            state.add_log(f"Error cleaning output directory: {e}")


task_manager = TaskManager()
