import json
class Memo():
	#just a bare class we can accumulate data in
	def serialize(self, path):
		try:
			with open(path + ".json","a+") as f:
				f.write(json.dumps(self.analysis))
				f.write("\n")
		except:
			#dont let this screw anything else up
			pass
		return self