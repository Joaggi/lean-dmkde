def initialization(experiment, local_path):
    import os 



    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__')

    try:  
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        import os
        os.system("pip3 install mlflow")
        drive_total_path = f'{local_path}{experiment}'
        os.chdir(drive_total_path)
        import sys
        sys.path = sys.path[1:]
        sys.path.insert(0, drive_total_path)
        sys.path.append('submodules/qmc/')
        sys.path.append('src/anomalydetection/')
        sys.path.append('data/')
        #sys.path.append('../../../../submodules/qmc/')
        print(sys.path)
        parent_path = drive_total_path



    elif is_interactive():   

        import sys
        parent_path = f"{local_path}{experiment}"
        # %cd ../../
        import os
        print(os.getcwd())
        sys.path.append(f"{parent_path}/submodules/qmc/")
        sys.path.append(f"{parent_path}/src/anomalydetection/")
        sys.path.append(f"{parent_path}/data/")
 
    else:

        #import pathlib
        #parent_dir = pathlib.Path(__file__).parent.parent.resolve() 
        #parent_path = str(parent_dir)


        parent_path = f"{local_path}{experiment}"
        import sys
        sys.path = sys.path[1:]

        os.chdir(parent_path)
        sys.path.insert(0, parent_path)
        sys.path.append(f'{parent_path}/submodules/qmc/')
        sys.path.append(f'{parent_path}/data/')
        sys.path.append(f'{parent_path}/src/anomalydetection/')
        #sys.path.append('../../../../submodules/qmc/')
        print(sys.path)
        # %cd ../../print(os.getcwd())


    return parent_path
