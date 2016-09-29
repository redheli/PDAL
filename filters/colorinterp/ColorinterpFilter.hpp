/******************************************************************************
* Copyright (c) 2016, Howard Butler, howard@hobu.co
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#pragma once

#include <pdal/plugin.hpp>
#include <pdal/Filter.hpp>

#include <gdal.h>
#include <ogr_spatialref.h>
#include <pdal/GDALUtils.hpp>

#include <map>

extern "C" int32_t ColorinterpFilter_ExitFunc();
extern "C" PF_ExitFunc ColorinterpFilter_InitPlugin();

namespace pdal
{

// Interpolates color ramp into Red, Green, and Blue dimensions
// for a given dimension
// specified dimensions. It also supports scaling the data by a multiplier
// on a per-dimension basis.
class PDAL_DLL ColorinterpFilter : public Filter
{
public:

    ColorinterpFilter()
        : m_interpDim(Dimension::Id::Z), m_min(0.0), m_max(0.0)
    {}
    ColorinterpFilter& operator=(const ColorinterpFilter&) = delete;
    ColorinterpFilter(const ColorinterpFilter&) = delete;

    static void * create();
    static int32_t destroy(void *);
    std::string getName() const;

private:
    virtual void addArgs(ProgramArgs& args);
    virtual bool processOne(PointRef& point);
    virtual void filter(PointView& view);


    Dimension::Id m_interpDim;
    double m_min;
    double m_max;
};

} // namespace pdal
