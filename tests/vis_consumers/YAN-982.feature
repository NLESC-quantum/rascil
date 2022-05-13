# Created by ord006 at 13/4/22
Feature:

  	Scenario: RCAL consumer can be called in receive workflow

		Given An example input file of the correct dimension
		And A receiver can be configured with a RCAL consumer
		And A scheduling block is available
		When the data is sent to the RCAL consumer
		Then The same data is received and written


