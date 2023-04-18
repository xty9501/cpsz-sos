#ifndef _IO_UTILS_HPP
#define _IO_UTILS_HPP

#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace IO{

template <class T>
void posix_write(std::string filename, T * data, size_t num_elements){
  int fd = open(filename.c_str(), O_CREAT|O_WRONLY|O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  write(fd, data, num_elements * sizeof(T));
  fsync(fd);
  close(fd);
}

template <class T>
void posix_read(std::string filename, T * data, size_t num_elements){
  int fd = open(filename.c_str(), O_RDONLY);
  read(fd, data, num_elements * sizeof(T));
  close(fd);
}

void posix_delete(std::string filename){
  unlink(filename.c_str());
}

void clear_cache(){
  std::ofstream fout("/proc/sys/vm/drop_caches", std::ios::out);
  int command = 3;
  fout << command;
  fout.close();
}

}
#endif