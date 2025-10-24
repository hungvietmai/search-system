"""
Async Indexing for Large-Scale Operations
Background task processing for feature extraction and indexing

Features:
- Async task queue
- Background indexing
- Progress tracking
- Retry logic
- Task status monitoring
"""
import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import logging
from queue import Queue
from threading import Thread
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IndexingTask:
    """Indexing task"""
    task_id: str
    task_type: str  # 'single', 'batch', 'directory'
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    error_message: Optional[str] = None
    result: Optional[Dict] = None
    
    @property
    def progress(self) -> float:
        """Calculate progress percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds"""
        if self.started_at is None:
            return None
        
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'progress': self.progress,
            'duration': self.duration,
            'error_message': self.error_message,
            'result': self.result
        }


class AsyncIndexer:
    """Async indexer for background processing"""
    
    def __init__(self,
                 feature_extractor,
                 milvus_client=None,
                 faiss_client=None,
                 max_workers: int = 2):
        """
        Initialize async indexer
        
        Args:
            feature_extractor: Feature extractor instance
            milvus_client: Milvus client instance
            faiss_client: Faiss client instance
            max_workers: Maximum concurrent workers
        """
        self.feature_extractor = feature_extractor
        self.milvus_client = milvus_client
        self.faiss_client = faiss_client
        self.max_workers = max_workers
        
        # Task management
        self.tasks: Dict[str, IndexingTask] = {}
        self.task_queue = Queue()
        self.workers: List[Thread] = []
        self.running = False
        
        logger.info(f"Async indexer initialized with {max_workers} workers")
    
    def start(self):
        """Start background workers"""
        if self.running:
            logger.warning("Indexer already running")
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} indexer workers")
    
    def stop(self):
        """Stop background workers"""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        logger.info("Stopped indexer workers")
    
    def _worker(self, worker_id: int):
        """Worker thread function"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from queue (with timeout)
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:
                    continue
                
                # Process task
                self._process_task(task, worker_id)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                # Queue get timeout or other error
                continue
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _process_task(self, task: IndexingTask, worker_id: int):
        """
        Process an indexing task
        
        Args:
            task: Indexing task
            worker_id: Worker ID
        """
        logger.info(f"Worker {worker_id} processing task {task.task_id}")
        
        # Update status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        
        try:
            if task.task_type == 'single':
                self._process_single_image(task)
            elif task.task_type == 'batch':
                self._process_batch(task)
            elif task.task_type == 'directory':
                self._process_directory(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now(timezone.utc)
    
    def _process_single_image(self, task: IndexingTask):
        """Process single image indexing"""
        assert task.result is not None, "Task result should not be None"
        image_path = task.result['image_path']
        file_id = task.result['file_id']
        
        # Extract features
        features = self.feature_extractor.extract_features(image_path)
        
        # Index in Milvus
        if self.milvus_client:
            milvus_id = self.milvus_client.insert_single(file_id, features)
            task.result['milvus_id'] = milvus_id
        
        # Index in Faiss
        if self.faiss_client:
            faiss_id = self.faiss_client.add_single(file_id, features)
            task.result['faiss_id'] = faiss_id
        
        task.processed_items = 1
    
    def _process_batch(self, task: IndexingTask):
        """Process batch indexing"""
        from app.batch_processor import BatchFeatureExtractor
        
        assert task.result is not None, "Task result should not be None"
        image_paths = task.result['image_paths']
        file_ids = task.result['file_ids']
        task.total_items = len(image_paths)
        
        # Batch feature extraction
        batch_processor = BatchFeatureExtractor(
            self.feature_extractor,
            batch_size=32
        )
        
        result = batch_processor.extract_batch(image_paths, file_ids)
        
        # Index in vector databases
        if result.success_count > 0:
            if self.milvus_client:
                self.milvus_client.insert_batch(result.file_ids, result.features)
            
            if self.faiss_client:
                self.faiss_client.add_batch(result.file_ids, result.features)
        
        task.processed_items = result.success_count
        task.failed_items = result.error_count
        task.result['errors'] = result.errors
    
    def _process_directory(self, task: IndexingTask):
        """Process directory indexing"""
        from app.batch_processor import BatchFeatureExtractor
        
        assert task.result is not None, "Task result should not be None"
        directory = Path(task.result['directory'])
        pattern = task.result.get('pattern', '*.jpg')
        
        # Find all images
        image_paths = list(directory.rglob(pattern))
        task.total_items = len(image_paths)
        
        if task.total_items == 0:
            logger.warning(f"No images found in {directory}")
            return
        
        # Process in batches
        batch_processor = BatchFeatureExtractor(
            self.feature_extractor,
            batch_size=32
        )
        
        result = batch_processor.extract_from_directory(
            directory,
            file_pattern=pattern,
            progress=False
        )
        
        # Index in vector databases
        if result.success_count > 0:
            if self.milvus_client:
                self.milvus_client.insert_batch(result.file_ids, result.features)
            
            if self.faiss_client:
                self.faiss_client.add_batch(result.file_ids, result.features)
        
        task.processed_items = result.success_count
        task.failed_items = result.error_count
        task.result['errors'] = result.errors
    
    def submit_single_image(self,
                           file_id: int,
                           image_path: str) -> str:
        """
        Submit single image for indexing
        
        Args:
            file_id: File ID
            image_path: Path to image
            
        Returns:
            Task ID
        """
        task = IndexingTask(
            task_id=str(uuid.uuid4()),
            task_type='single',
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            total_items=1,
            result={
                'file_id': file_id,
                'image_path': image_path
            }
        )
        
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        
        logger.info(f"Submitted single image task {task.task_id}")
        return task.task_id
    
    def submit_batch(self,
                    file_ids: List[int],
                    image_paths: List[str]) -> str:
        """
        Submit batch for indexing
        
        Args:
            file_ids: List of file IDs
            image_paths: List of image paths
            
        Returns:
            Task ID
        """
        task = IndexingTask(
            task_id=str(uuid.uuid4()),
            task_type='batch',
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            total_items=len(file_ids),
            result={
                'file_ids': file_ids,
                'image_paths': image_paths
            }
        )
        
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        
        logger.info(f"Submitted batch task {task.task_id} ({len(file_ids)} images)")
        return task.task_id
    
    def submit_directory(self,
                        directory: str,
                        pattern: str = '*.jpg') -> str:
        """
        Submit directory for indexing
        
        Args:
            directory: Directory path
            pattern: File pattern
            
        Returns:
            Task ID
        """
        task = IndexingTask(
            task_id=str(uuid.uuid4()),
            task_type='directory',
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            result={
                'directory': directory,
                'pattern': pattern
            }
        )
        
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        
        logger.info(f"Submitted directory task {task.task_id}: {directory}")
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Get status of a task
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dict or None
        """
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks"""
        return [task.to_dict() for task in self.tasks.values()]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task (only if pending)
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False otherwise
        """
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
        return False
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.task_queue.qsize()
    
    def get_stats(self) -> Dict:
        """Get indexer statistics"""
        total_tasks = len(self.tasks)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            'total_tasks': total_tasks,
            'pending_tasks': pending,
            'running_tasks': running,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'queue_size': self.get_queue_size(),
            'workers': len(self.workers),
            'running': self.running
        }


# Global indexer instance
_async_indexer = None


def get_async_indexer(feature_extractor=None,
                      milvus_client=None,
                      faiss_client=None) -> AsyncIndexer:
    """
    Get or create global async indexer
    
    Args:
        feature_extractor: Feature extractor instance
        milvus_client: Milvus client
        faiss_client: Faiss client
        
    Returns:
        AsyncIndexer instance
    """
    global _async_indexer
    
    if _async_indexer is None:
        if feature_extractor is None:
            from app.feature_extractor import get_feature_extractor
            feature_extractor = get_feature_extractor()
        
        if milvus_client is None:
            from app.milvus_client import get_milvus_client
            milvus_client = get_milvus_client()
        
        if faiss_client is None:
            from app.faiss_client import get_faiss_client
            faiss_client = get_faiss_client()
        
        _async_indexer = AsyncIndexer(
            feature_extractor,
            milvus_client,
            faiss_client,
            max_workers=2
        )
        
        # Start workers
        _async_indexer.start()
    
    return _async_indexer

