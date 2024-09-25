# coding=utf-8
'''
Created on Jul 12, 2018

@author: adapa
'''

import os

from sqlobject.col import DateCol, ForeignKey, StringCol
from sqlobject.dbconnection import connectionForURI
from sqlobject.main import SQLObject, sqlhub

from PyLOSt.util.resource_path import resource_path


##################Database section##################
def connectDB(db_loc = resource_path(os.path.join('databases', 'gs.db'))):
    try:
        db_filename = os.path.abspath(db_loc)
        sqlhub.processConnection = connectionForURI('sqlite:'+db_filename)
        return sqlhub.processConnection
    except Exception as e:
        print('Database connection failed')
        print(e)
    return None
##################Database section##################

class StitchSetupOptionsInstr(SQLObject):
    instr               = ForeignKey('Instruments') 
    option              = ForeignKey('StitchSetupOptionsCommon')
    defVal              = StringCol() 
    defValUnit          = StringCol() 
    addedBy             = StringCol()
    location            = ForeignKey('Locations')
    dateAdded           = DateCol()
    
class StitchSetupOptionsCommon(SQLObject):
    option              = StringCol(alternateID=True)
    optionDesc          = StringCol()
    dispType            = ForeignKey('InputDispTypes')
    defVal              = StringCol() 
    defValUnit          = StringCol() 
    addedBy             = StringCol()
    location            = ForeignKey('Locations')
    dateAdded           = DateCol()
    
class StitchSetupAlgoOptions(SQLObject):
    algo                = ForeignKey('Algorithms')
    option              = StringCol()
    optionDesc          = StringCol() 
    dispType            = ForeignKey('InputDispTypes')
    defVal              = StringCol() 
    allVals             = StringCol() 
    valUnit             = StringCol()
    groupItems          = StringCol()
    addedBy             = StringCol()
    dateAdded           = DateCol()
    
class Instruments(SQLObject):
    instrId             = StringCol(alternateID=True)
    instrName           = StringCol() 
    instrType           = ForeignKey('InstrTypes') 
    instrLocation       = ForeignKey('Locations')
    dataFormats         = StringCol() 
    addedBy             = StringCol()
    dateAdded           = DateCol()
    
class Algorithms(SQLObject):
    algoName            = StringCol(alternateID=True)
    algoDesc            = StringCol() 
    algoType            = ForeignKey('AlgoTypes')
    functionName        = StringCol() 
    addedBy             = StringCol()
    location            = ForeignKey('Locations')
    dateAdded           = DateCol()
    
class AppInfo(SQLObject):
    appName         = StringCol()
    appLanguage     = StringCol() #python
    version         = StringCol()
    createdBy       = StringCol()
    location        = ForeignKey('Locations')
    dateCreated     = DateCol()
    dateModified    = DateCol()
    
class Locations(SQLObject):
    location           = StringCol(alternateID=True)
    locationDesc       = StringCol() 
    
class AlgoTypes(SQLObject):
    algoType           = StringCol(alternateID=True)
    algoTypeDesc       = StringCol() 
    
class InstrTypes(SQLObject):
    instrType          = StringCol(alternateID=True)
    instrDesc          = StringCol()
    
class InputDispTypes(SQLObject):
    dispType           = StringCol(alternateID=True)
    dispDesc           = StringCol()

class ConfigParams(SQLObject):
    paramName          = StringCol(alternateID=False)
    paramDesc          = StringCol()
    paramType          = StringCol() # S:static, D:dynamic
    paramValue         = StringCol()
    dateCreated        = DateCol()

class InstrTypeFunctionMap(SQLObject):
    instrType          = ForeignKey('InstrTypes')
    dataFormat         = StringCol()
    fileName           = StringCol()
    className          = StringCol()
    dateCreated        = DateCol()

class StitchViewerButtons(SQLObject):
    name               = StringCol()
    description        = StringCol()
    fileName           = StringCol()
    className          = StringCol()
    type               = ForeignKey('StitchViewerButtonTypes')
    requires           = StringCol()
    requiresText       = StringCol()
    dateCreated        = DateCol()

class StitchViewerButtonTypes(SQLObject):
    type               = StringCol()
    typeDesc           = StringCol()

if __name__ == "__main__":
    connectDB(db_loc='gs.db')
#     InstrTypes.dropTable(ifExists=True)
#     AlgoTypes.dropTable(ifExists=True)
#     Locations.dropTable(ifExists=True)
#     AppInfo.dropTable(ifExists=True)
#     Algorithms.dropTable(ifExists=True)
#     Instruments.dropTable(ifExists=True)
#     StitchSetupOptionsCommon.dropTable(ifExists=True)
#     StitchSetupOptionsInstr.dropTable(ifExists=True)  
#     InputDispTypes.dropTable(ifExists=True)
#     StitchSetupAlgoOptions.dropTable(ifExists=True)
    
#     InstrTypes.createTable(ifNotExists=True)
#     AlgoTypes.createTable(ifNotExists=True)
#     Locations.createTable(ifNotExists=True)
#     AppInfo.createTable(ifNotExists=True)
#     Algorithms.createTable(ifNotExists=True)
#     Instruments.createTable(ifNotExists=True)
#     StitchSetupOptionsCommon.createTable(ifNotExists=True)
#     StitchSetupOptionsInstr.createTable(ifNotExists=True)
#     
#     # static data in the tables
#     InstrTypes(instrType='SHP',instrDesc='SHARPeR')
#     InstrTypes(instrType='FIZ',instrDesc='Fizeau interferometer')
#     InstrTypes(instrType='MSI',instrDesc='Micro stitching interferometer')
#     
#     AlgoTypes(algoType='PR',algoTypeDesc='Pre Processing')
#     AlgoTypes(algoType='S',algoTypeDesc='Stitching')
#     AlgoTypes(algoType='PS',algoTypeDesc='Post Processing')

#     InputDispTypes.createTable(ifNotExists=True)
#     StitchSetupAlgoOptions.createTable(ifNotExists=True) #also create unique key for algo+option combined
#     
#     InputDispTypes(dispType='B',dispDesc='Button')
#     InputDispTypes(dispType='E',dispDesc='LineEdit')
#     InputDispTypes(dispType='C',dispDesc='Checkbox')
#     InputDispTypes(dispType='S',dispDesc='Select box')
#     InputDispTypes(dispType='R',dispDesc='Radio Button')
#     InputDispTypes(dispType='TE',dispDesc='TextEdit')
#     InputDispTypes(dispType='DE',dispDesc='DateEdit')
#     InputDispTypes(dispType='DTE',dispDesc='DateTimeEdit')
#     InputDispTypes(dispType='L',dispDesc='ListWidget')

    # ConfigParams.dropTable(ifExists=True)
    # ConfigParams.createTable(ifNotExists=True)
    # ConfigParams(paramName='LAST_H5_LOC',paramDesc='Last opened h5 file directory', paramType='D', paramValue='', dateCreated=datetime.today().strftime('%Y-%m-%d'))

    # InstrTypeFunctionMap.dropTable(ifExists=True)
    # InstrTypeFunctionMap.createTable(ifNotExists=True)
    # InstrTypeFunctionMap(instrType=1,dataFormat='has', fileName='PyLOSt.data_in.sharper.has_to_h5', className='HasToH5', dateCreated=datetime.today().strftime('%Y-%m-%d'))
    # InstrTypeFunctionMap(instrType=2,dataFormat='dat', fileName='PyLOSt.data_in.fizeau.dat_to_h5', className='DatToH5', dateCreated=datetime.today().strftime('%Y-%m-%d'))
    # InstrTypeFunctionMap(instrType=2,dataFormat='datx', fileName='PyLOSt.data_in.fizeau.datx.datx_to_h5', className='DatxToH5', dateCreated=datetime.today().strftime('%Y-%m-%d'))
    # InstrTypeFunctionMap(instrType=3,dataFormat='OPD', fileName='PyLOSt.data_in.msi.opd_to_h5', className='OpdToH5', dateCreated=datetime.today().strftime('%Y-%m-%d'))
    # InstrTypeFunctionMap(instrType=5,dataFormat='slp2', fileName='PyLOSt.data_in.ltp.slp2_to_h5', className='Slp2ToH5', dateCreated=datetime.today().strftime('%Y-%m-%d'))

    # StitchViewerButtons.dropTable(ifExists=True)
    # StitchViewerButtons.createTable(ifNotExists=True)

    # StitchViewerButtonTypes.dropTable(ifExists=True)
    # StitchViewerButtonTypes.createTable(ifNotExists=True)
    # StitchViewerButtonTypes(type='S',typeDesc='Single use')
    # StitchViewerButtonTypes(type='R',typeDesc='Repeat use')
    # StitchViewerButtonTypes(type='SD',typeDesc='Single use, default sequence')
    # StitchViewerButtonTypes(type='RD',typeDesc='Repeat use, default sequence')
