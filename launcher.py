import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    import app as app_module
    app_module.init_cluster()

    import uvicorn
    uvicorn.run(app_module.app, host='0.0.0.0', port=8000, log_level='info')
