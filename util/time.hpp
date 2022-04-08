#ifndef TIME
#define TIME

#include <sys/time.h>

inline struct timeval GetTimeStamp() {
	struct timeval tv;
   gettimeofday(&tv,__null);
   return tv;
}

#endif