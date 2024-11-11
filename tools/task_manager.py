import threading
from queue import Queue
from enum import Enum
from typing import Callable, Any, Dict, Optional
import time

class TaskStatus(Enum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    FAIL = 'FAIL'

class TaskManager:
    def __init__(self):
        self.tasks = Queue()
        self.task_status: Dict[str, TaskStatus] = {}
        self.current_task_id = ''
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run_tasks)
        self.thread.daemon = True
        self.thread.start()

    def add_task(self, task_id: str, task: Callable[[], Any]):
        """
        添加任务到队列中。
        :param task_id: 任务的唯一标识符
        :param task: 任务函数，无参数，返回任何类型的结果
        """
        with self.lock:
            if task_id in self.task_status:
                raise ValueError(f"Task with ID {task_id} already exists.")
            self.tasks.put((task_id, task))
            self.task_status[task_id] = TaskStatus.PENDING
            print(f"Task {task_id} added to the queue.")
            return task_id

    def delete_task(self, task_id: str):
        """
        删除任务。
        :param task_id: 任务的唯一标识符
        """
        with self.lock:
            if task_id not in self.task_status:
                raise ValueError(f"Task with ID {task_id} does not exist.")
            if self.task_status[task_id] == TaskStatus.RUNNING:
                raise ValueError(f"Cannot delete task {task_id} as it is currently running.")
            self.task_status.pop(task_id)
            new_tasks = Queue()
            while not self.tasks.empty():
                t_id, task = self.tasks.get()
                if t_id != task_id:
                    new_tasks.put((t_id, task))
            self.tasks = new_tasks
            print(f"Task {task_id} deleted from the queue.")

    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        获取任务的状态。
        :param task_id: 任务的唯一标识符
        :return: 任务的状态
        """
        with self.lock:
            if task_id not in self.task_status:
                raise ValueError(f"Task with ID {task_id} does not exist.")
            return self.task_status[task_id]

    def _run_tasks(self):
        """
        内部方法，用于在单独的线程中运行任务队列中的任务。
        """
        while True:
            with self.lock:
                if not self.tasks.empty():
                    task_id, task = self.tasks.get()
                    self.current_task_id = task_id
                    self.task_status[task_id] = TaskStatus.RUNNING
                    print(f"Starting task {task_id}")
                    try:
                        result = task()
                        if result.code == 200:
                            print(f"Task {task_id} completed successfully. Result: {result}")
                            self.task_status[task_id] = TaskStatus.SUCCESS
                        else:
                            print(f"Task {task_id} failed. Result: {result}")
                            self.task_status[task_id] = TaskStatus.FAIL
                    except Exception as e:
                        print(f"Task {task_id} failed with exception: {e}")
                        self.task_status[task_id] = TaskStatus.FAIL
                    finally:
                        self.current_task_id = ''
                        self.tasks.task_done()
                else:
                    # 如果队列为空，让线程短暂休眠以减少CPU占用
                    threading.Event().wait(0.1)

# 示例任务函数
def example_task():
    import time
    print("Task is running...")
    time.sleep(5)
    print("Task completed.")
    return "Task result"

# 使用示例
if __name__ == "__main__":
    task_manager = TaskManager()
    
    # 添加任务
    task_manager.add_task("task1", example_task)
    task_manager.add_task("task2", example_task)
    task_manager.add_task("task3", example_task)
    
    # 获取任务状态
    print(task_manager.get_task_status("task1"))
    
    # 删除任务
    task_manager.delete_task("task2")
    
    # 主线程继续运行，任务在后台按顺序执行
    while True:
        time.sleep(1)