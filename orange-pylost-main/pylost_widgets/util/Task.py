# coding=utf-8
import concurrent


class Task:
    """Task class for threading as implemented in Orange"""

    future = ...  # type: concurrent.futures.Future

    watcher = ...  # type: FutureWatcher

    cancelled = False  # type: bool

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        # cancel the future. Note this succeeds only if the execution has
        # not yet started (see `concurrent.futures.Future.cancel`) ..
        self.future.cancel()
        # ... and wait until computation finishes
        concurrent.futures.wait([self.future])
