/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef UsecTimer_h
#define UsecTimer_h

#include <time.h>

#include "timers.h"

using namespace std;

#ifdef __APPLE__ ///////////////////////////////////////////////

class UsecTimer {
public:
    typedef double Time;
    
    UsecTimer(string description){
        mDescription = description;
        mCumulativeElapsedSeconds = 0.0;
        mNumSamples = 0;
        mStarted = false;
    }
    ~UsecTimer(){}
    void start(){
      mStart = Realm::Clock::current_time();
        mStarted = true;
    }
    void stop(){
        if(mStarted) {
          Time end = Realm::Clock::current_time();
            mCumulativeElapsedSeconds += end - mStart;
            mNumSamples++;
            mStarted = false;
        }
    }
    string to_string(){
        double meanSampleElapsedSeconds = 0;
        if(mNumSamples > 0) {
            meanSampleElapsedSeconds = mCumulativeElapsedSeconds / mNumSamples;
        }
        double sToUs = 1000000.0;
        std::ostringstream output;
        output << mDescription
        << " " << (mCumulativeElapsedSeconds) << " sec"
        << " " << (mCumulativeElapsedSeconds * sToUs)
        << " usec = " << (meanSampleElapsedSeconds * sToUs)
        << " usec * " << (mNumSamples)
        << (mNumSamples == 1 ? " sample" : " samples");
        return output.str();
    }
    
private:
    static double timespecToSeconds(timespec *t) {
        const double nsecToS = 1.0 / 1000000000.0;
        return (double)t->tv_sec + (double)t->tv_nsec * nsecToS;
    }
    
    bool mStarted;
    Time mStart;
    string mDescription;
    double mCumulativeElapsedSeconds;
    int mNumSamples;
};


#else ///////////////////////////////////////////////

class UsecTimer {
public:
    typedef struct timespec Time;
    static const clockid_t CLOCK = CLOCK_MONOTONIC;
    
    UsecTimer(string description){
        mDescription = description;
        mCumulativeElapsedSeconds = 0.0;
        mNumSamples = 0;
        mStarted = false;
    }
    ~UsecTimer(){}
    void start(){
        if(clock_gettime(CLOCK, &mStart)) {
            cerr << "error from clock_gettime" << endl;
            return;
        }
        mStarted = true;
    }
    void stop(){
        if(mStarted) {
            Time end;
            if(clock_gettime(CLOCK, &end)) {
                cerr << "error from clock_gettime" << endl;
                return;
            }
            double elapsedSeconds = timespecToSeconds(&end) - timespecToSeconds(&mStart);
            mCumulativeElapsedSeconds += elapsedSeconds;
            mNumSamples++;
            mStarted = false;
        }
    }
    string to_string(){
        double meanSampleElapsedSeconds = 0;
        if(mNumSamples > 0) {
            meanSampleElapsedSeconds = mCumulativeElapsedSeconds / mNumSamples;
        }
        double sToUs = 1000000.0;
        std::ostringstream output;
        output << mDescription
        << " " << (mCumulativeElapsedSeconds) << " sec"
        << " " << (mCumulativeElapsedSeconds * sToUs)
        << " usec = " << (meanSampleElapsedSeconds * sToUs)
        << " usec * " << (mNumSamples)
        << (mNumSamples == 1 ? " sample" : " samples");
        return output.str();
    }
    
private:
    static double timespecToSeconds(timespec *t) {
        const double nsecToS = 1.0 / 1000000000.0;
        return (double)t->tv_sec + (double)t->tv_nsec * nsecToS;
    }
    
    bool mStarted;
    Time mStart;
    string mDescription;
    double mCumulativeElapsedSeconds;
    int mNumSamples;
};

#endif ///////////////////////////////////////////////


#endif /* UsecTimer_h */
