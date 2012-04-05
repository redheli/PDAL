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

#include <iostream>

#include <pdal/PipelineReader.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PipelineWriter.hpp>
#include <pdal/FileUtils.hpp>

#include "AppSupport.hpp"

#include "Application.hpp"

#include <iostream>

using namespace pdal;
namespace po = boost::program_options;


class PcPipeline : public Application
{
public:
    PcPipeline(int argc, char* argv[]);
    int execute();

private:
    void addSwitches();
    void validateSwitches();

    std::string m_inputFile;
    std::string m_pipelineFile;
    bool m_usestdin;
};


PcPipeline::PcPipeline(int argc, char* argv[])
    : Application(argc, argv, "pcpipeline")
    , m_inputFile("")
{
    return;
}


void PcPipeline::validateSwitches()
{
    if (m_inputFile == "" && ! m_usestdin)
    {
        throw app_usage_error("input file name required");
    }

    return;
}


void PcPipeline::addSwitches()
{
    po::options_description* file_options = new po::options_description("file options");

    file_options->add_options()
        ("input,i", po::value<std::string>(&m_inputFile)->default_value(""), "input file name")
        ("pipeline-serialization", po::value<std::string>(&m_pipelineFile)->default_value(""), "")
        ("stdin,s", po::value<bool>(&m_usestdin)->zero_tokens()->implicit_value(true), "Read pipeline XML from stdin")
        ;

    addSwitchSet(file_options);
    addPositionalSwitch("input", 1);
}


int PcPipeline::execute()
{
    if (!FileUtils::fileExists(m_inputFile) && !m_usestdin)
    {
        throw app_runtime_error("file not found: " + m_inputFile);
    }

    pdal::PipelineManager manager;

    pdal::PipelineReader reader(manager, isDebug(), getVerboseLevel());
    bool isWriter(false);
    
    if (!m_usestdin)
        isWriter = reader.readPipeline(m_inputFile);
    else
        isWriter = reader.readPipeline(std::cin);
        
    if (!isWriter)
        throw app_runtime_error("pipeline file is not a Writer");

    const boost::uint64_t np = manager.execute();

    std::cout << "Wrote " << np << " points.\n";
    
    if (m_pipelineFile.size() > 0)
    {
        pdal::PipelineWriter writer(manager);
        writer.writePipeline(m_pipelineFile);
    }
    return 0;
}


int main(int argc, char* argv[])
{
    PcPipeline app(argc, argv);
    return app.run();
}



