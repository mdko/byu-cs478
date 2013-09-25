// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef ERROR_H
#define ERROR_H

#include <string>
#include <sstream>

void AssertFailed();

// This macro is for checking values at run-time, but only in debug mode.
#ifdef _DEBUG
#  ifdef WIN32
#    define Assert(x)			\
	{				\
		if(!(x))		\
		{			\
			AssertFailed(); \
			__asm int 3	\
		}			\
	}
#  else // WIN32
#    define Assert(x)			\
	{				\
		if(!(x))                \
			AssertFailed(); \
	}
#  endif // !WIN32
#else // _DEBUG
#  define Assert(x)	((void)0)
#endif // else _DEBUG


// This macro is for checking things at compile-time
#define COMPILER_ASSERT(expr)  enum { CompilerAssertAtLine##__LINE__ = sizeof( char[(expr) ? +1 : -1] ) }


// Convert any type to a string
template<typename T>
std::string to_str(const T& n)
{
	std::ostringstream os;
	os.precision(14);
	os << n;
	return os.str();
}


// These methods throw an exception. If you always use these methods instead
// of calling "throw" yourself, then you will always have a central location
// where you can put a break-point to stop before an exception is thrown.
void ThrowError(std::string s1);
void ThrowError(std::string s1, std::string s2);
void ThrowError(std::string s1, std::string s2, std::string s3);
void ThrowError(std::string s1, std::string s2, std::string s3, std::string s4);
void ThrowError(std::string s1, std::string s2, std::string s3, std::string s4, std::string s5);
void ThrowError(std::string s1, std::string s2, std::string s3, std::string s4, std::string s5, std::string s6);
void ThrowError(std::string s1, std::string s2, std::string s3, std::string s4, std::string s5, std::string s6, std::string s7);


#ifdef WIN32
// Windows seems to be missing these POSIX standard definitions
# ifndef M_PI
#  define M_E            2.7182818284590452354   /* e */
#  define M_LOG2E        1.4426950408889634074   /* log_2 e = 1/(log_e 2)*/
#  define M_LOG10E       0.43429448190325182765  /* log_10 e = 1/(log_e 10)*/
#  define M_LN2          0.69314718055994530942  /* log_e 2 */
#  define M_LN10         2.30258509299404568402  /* log_e 10 */
#  define M_PI           3.14159265358979323846  /* pi */
#  define M_PI_2         1.57079632679489661923  /* pi/2 */
#  define M_PI_4         0.78539816339744830962  /* pi/4 */
#  define M_1_PI         0.31830988618379067154  /* 1/pi */
#  define M_2_PI         0.63661977236758134308  /* 2/pi */
#  define M_2_SQRTPI     1.12837916709551257390  /* 2/sqrt(pi) */
#  define M_SQRT2        1.41421356237309504880  /* sqrt(2) */
#  define M_SQRT1_2      0.70710678118654752440  /* 1/sqrt(2) */
# endif
#endif // WIN32


#ifdef WIN32
# define NOMINMAX
# undef min
# undef max
#endif


#endif // ERROR_H
