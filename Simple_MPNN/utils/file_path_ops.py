import os
import shutil
import glob
import fnmatch
import shutil
from pprint import pprint


class CustomPath():
    '''Purpose: File Manipulation'''

    def __init__(self, path):
        self.path = path


    def get_file_paths(self, pattern='*.*', recursive=True, sorted_asc=True):
        '''To get all file names within a given path those names match a given pattern

        :param pattern: specified pattern (return files whose names match this pattern)
            Eg. pattern='*.*'   : all files
                pattern='*.fcs" : only fcs files (Note need to have * )
                Note: fnmatch lib is file name match with Unix shell-style wildcards
                      (NOT regular expression as in re lib)
        :param recursive: if True run recursively within all sub-directories else only in given directory
        :param sorted_asc: if True sorted file path in ascending order else keep the original order
        :return:
            file_paths: a list of file path
        '''
        if recursive:
            file_paths = []
            for root, dir, filenames in os.walk(self.path):
                for filename in fnmatch.filter(filenames, pattern):
                    file_paths.append(os.path.join(root, filename))
        else:
            file_paths = glob.glob(os.path.join(self.path, pattern))

        if sorted_asc:
            file_paths.sort()

        return file_paths


    def get_file_paths_with_size(self, pattern='*.*', recursive=True, sorted_asc=True, unit='MB'):
        ''' To summarize of folder size, number of files and size of each file
            in the folder.
        Args:
        -----
        Input
        self.path: a path folder
        unit: unit of file size.
        Output:
        summary: size of folder and number of files in this folder
        dic_fp_size: key is file path and value is its size
        '''
        dic = {'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4}
        unit = unit.upper()
        unit = 'Byte' if unit not in dic else unit
        size_path = 0
        dic_fp_size = {}

        file_paths = self.get_file_paths(pattern, recursive, sorted_asc)
        for file_path in file_paths:
            size = os.path.getsize(file_path)  # Unit: Bytes
            if unit in dic:
                size /= dic[unit]
            ## End of if
            dic_fp_size[file_path] = str(round(size, 2)) + f' {unit}'
            size_path += size
        # End of for root,...
        summary = str(round(size_path, 2)) + f' {unit} of {len(dic_fp_size)} files'
        return summary, dic_fp_size


    def get_path(self, new_folder=None):
        '''To get new path:
        If it is NOT existed, create a new folder in self.path and return this new path.
        Otherwise return its existed path.
        '''
        if new_folder:
            new_path = os.path.join(self.path, new_folder)
        else:
            new_path = self.path

        try:
            os.mkdir(new_path)
        except:  # existed
            pass
        return new_path


    def delete_this_path(self):
        print('Be careful - DELETE all files and its folder name')
        print(f'folder = {self.path}')
        answer = input('Delete this folder: yes or no? ')
        if answer.lower() == 'yes':
            try:
                print('Deleting...')
                shutil.rmtree(self.path)
                print('Completely deleted')
            except FileNotFoundError:
                print(f"NOT found: file path = {self.path}")
                print('You may need to choose another path')
        else:
            print('You choose: Not delete this folder')


    def delete_this_path_WITHOUT_confirm(self):
        '''Be careful - DELETE all files and its folder name
           ONLY use in automation when surely know what will be deleted.
        '''
        print(f'folder = {self.path}')
        try:
            print('Deleting...')
            shutil.rmtree(self.path)
            print('Completely Deleted')
        except FileNotFoundError:
            print(f"NOT found: file path = {self.path}")
            print('You may need to choose another path')


    def copy_files_to_destination(self, pattern='*.*', recursive=True, sorted_asc=True, destination=None):
        if not destination:
            return None
        _file_paths = self.get_file_paths(pattern, recursive, sorted_asc)
        for _file_path in _file_paths:
            shutil.copy(_file_path, destination)
        return None


    def get_one_level_subdirectories(self):
        '''To get one_level_subdirectories only - Not recursive in lower subdirectories
        :return: a list of all subdirectories
        '''
        _path = self.path + '\*'  #
        return glob.glob(_path)



def open_excel(path=r'C:\DB\FACS files', pattern='*.xlsx'):
    '''
    :param path:
    :param pattern:
    :return:
    '''
    excel_file_paths = CustomPath(path).get_file_paths(pattern=pattern)
    print(f'Number of Excel files = {len(excel_file_paths)}')
    for f_p in excel_file_paths:
        try:
            print(f'open file = {f_p}')
            os.startfile(f_p)
        except:
            print(f'Cannot open file = {f_p}')
    print('End of opening Excel files')


def test_class_CustomPath(path=r'C:\DB\FACS files', pattern='*.fcs'):
    '''
    :param path:
    :return:
    Output example:
    summary = 113136879.19 KB of 6651 files
    C:\DB\FACS files\2015_12\APC 20151209\Specimen_001_Percp.fcs
    C:\DB\FACS files\2015_12\B Cell 20151209\Specimen_001_607.fcs
    ....
    C:\DB\FACS files\CPI_310118\PBMCs_Th GEM 152 18jan18_059.fcs :  476.58 KB
    C:\DB\FACS files\CPI_310118\PBMCs_Th HBD63 7march17_038.fcs :  26328.15 KB
    '''
    my_path = CustomPath(path)

    file_paths_w_pattern = my_path.get_file_paths(pattern=pattern)
    print('List of file path')
    for f_p in file_paths_w_pattern:
        print(f_p)

    summary, dic_fp_size = my_path.get_file_paths_with_size(pattern=pattern, unit='KB')
    print('all_file_paths_w_size')
    for fp, size in dic_fp_size.items():
        print(fp, ': ', size)
    print(f'summary = {summary}')

    print('get_path: existed path')
    this_path = my_path.get_path()
    print('this path = ', this_path)

    print('get_path: make new path')
    my_new_path = my_path.get_path(new_folder='new_folder_for_test')
    print('my_new_path = ', my_new_path)

    # delete path: tested with other path
    return None


if __name__ == '__main__':
    pass