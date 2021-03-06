/******************************************************************************
* Copyright (c) 2011, Michael P. Gerlek (mpg@flaxen.com)
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

#include <memory>

extern "C" int32_t ReprojectionFilter_ExitFunc();
extern "C" PF_ExitFunc ReprojectionFilter_InitPlugin();

namespace pdal
{

namespace gdal
{
    class ErrorHandler;
}

class PDAL_DLL ReprojectionFilter : public Filter
{
public:
    ReprojectionFilter();
    ~ReprojectionFilter();

    static void * create();
    static int32_t destroy(void *);
    std::string getName() const;

private:
    virtual void addArgs(ProgramArgs& args);
    virtual void initialize();
    virtual void ready(PointTableRef table);
    virtual PointViewSet run(PointViewPtr view);
    virtual bool processOne(PointRef& point);

    void updateBounds();
    void createTransform(const SpatialReference& srs);

    SpatialReference m_inSRS;
    SpatialReference m_outSRS;
    bool m_inferInputSRS;

    typedef void* ReferencePtr;
    typedef void* TransformPtr;
    ReferencePtr m_in_ref_ptr;
    ReferencePtr m_out_ref_ptr;
    TransformPtr m_transform_ptr;
    gdal::ErrorHandler* m_errorHandler;

    ReprojectionFilter& operator=(const ReprojectionFilter&); // not implemented
    ReprojectionFilter(const ReprojectionFilter&); // not implemented
};

} // namespace pdal
